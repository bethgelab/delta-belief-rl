
import numpy as np
from sympy.logic import false
from tqdm import tqdm
import copy
from collections import defaultdict
from typing import Union, List, Dict, Any, Tuple
from torch.optim._multi_tensor import partialclass
from delta_belief_rl.llm_agent.generation import LLMGenerationManager
import verl.utils.torch_functional as verl_F
from verl import DataProto
from delta_belief_rl.utils.format import pad_dataproto_to_divisor, unpad_dataproto
import re
from delta_belief_rl.llm_agent.prompts import get_elicitation


class BeliefManager(LLMGenerationManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def belief_logprob(self, raw_prompt: np.ndarray, gts: np.ndarray, game_status: np.ndarray, calculate_entropy:bool=False, return_vocab:bool=False) -> dict:
        """
        Estimate the beliefs for pre-generated trajectories.

        Args:
            raw_prompt: np.ndarray the raw prompt for the pre-generated trajectories
            gts: np.ndarray - the ground truth secrets
            n: int - the number of trajectories to estimate the beliefs for
            
        """

        self.history = [copy.deepcopy(d) for d in raw_prompt.tolist()]

        #create a dictionary to store the belief results
        belief_results = {}
        for idx, gt_secret in enumerate(gts):
            belief_results[str(idx)] = {
                "ground_truth": gt_secret,
                "messages": self.history[idx],
                "logprob": [],
                "diff_logprob": [],
                "entropy": [],
                "vocab": [],
                "game_status": None
            }
            if game_status is not None:
                belief_results[str(idx)]["game_status"] = game_status[idx]

        
        #longest trajectory length
        max_length = max([len(hist) for hist in self.history])
        if max_length % 2 == 0:
            max_steps = max_length // 2
        else:
            max_steps = max_length // 2 
            max_steps += 1
        end = 0
        active = [1 for i in range(len(self.history))]
        for step in tqdm(range(max_steps), desc="Belief Estimation Steps"):
            end += 2
            messages = []
            gt_active = []
            for idx, (hist, gt) in enumerate(zip(self.history, gts)):
                #get out the path trajectory
                if len(hist) >= end:
                    # select the last history side
                    current_hist = copy.deepcopy(hist[:end])
                    current_hist.append({"role": "assistant", "content": get_elicitation(gt)})
                    messages.append(current_hist)
                    gt_active.append(gt)
                else:
                    active[idx] = 0 


            if not messages:
                break
            
            #now format as token ids 
            rollings = self._format_input(messages, generation_prompt=False) 
            
            #now create the responses subset and responses mask of the gt 
            responses, responses_mask = self._create_responses_mask(messages, gt_active, self.tokenizer_actor)

            rollings_responses = DataProto.from_dict({"responses":responses, "responses_attention_mask": responses_mask})
            

            #add to rollings
            rollings = rollings.union(rollings_responses)

            #add gt for debugging
            rollings.meta_info['ground_truth'] = np.array(gt_active)

            if self.config.debug:
                print(f'[DEBUG] messages: {messages[0][-1]}')
                print(f'[DEBUG] rollings: {rollings.batch["input_ids"][0,-responses_mask.shape[1]:]}')
                print(f'[DEBUG] rollings mask: {responses_mask[0]}')
                str_mask = self.tokenizer.decode(
                    rollings.batch['input_ids'][0,-responses_mask.shape[1]:][responses_mask[0] == 1], 
                    skip_special_tokens=False)
                print(f'[DEBUG] mask corresponding to token: {str_mask}')

            #logprob
            rollings_padded, pad_size = pad_dataproto_to_divisor(rollings, self.actor_rollout_wg.world_size) #bsz, prompt_len
            #adjust meta_info to adjust logprob calculation parameters
            rollings_padded.meta_info["logprob_secret"] = {"temperature": 1, "calculate_entropy": False}
            output_padded = self.actor_rollout_wg.compute_log_prob(rollings_padded) # bsz, responses_len
            output = unpad_dataproto(output_padded, pad_size=pad_size)

            #log_prob 
            logprob_agg = self.get_scores(output.batch['log_probs'], responses_mask, gt_active, responses) #bsz,
            assert len(gt_active) == logprob_agg.shape[0], f"gt_active and logprob_mean must have the same length, got {len(gt_active)} and {logprob_agg.shape[0]}"
            assert sum(active) == len(logprob_agg), f'active must match the number of logprobs'

            # entropy for the secret word 
            if calculate_entropy:
                entropy_secrets = self.extract_per_secret(output.batch['entropy'], responses_mask, gt_active) #bsz,
                assert len(gt_active) == entropy_secrets.shape[0], f"gt_active and entropy must have the same length, got {len(gt_active)} and {entropy_secrets.shape[0]}"
            
            
            #vocabulary distribution 
            if return_vocab:
                # vocab_dist_secrets = self.extract_per_secret(output.batch['vocab'], responses_mask, gt_active) #bsz,
                vocab_logprob_secrets  = output.batch['vocab']
                assert len(gt_active) == vocab_logprob_secrets.shape[0], f"gt_active and vocab must have the same length, got {len(gt_active)} and {vocab_lobprob_secrets.shape[0]}"


            #store the logprobs for each secret
            idx_active = 0
            for idx, active_status in enumerate(active):
                if active_status:
                    lp = logprob_agg[idx_active]
                    belief_results[str(idx)]['logprob'].append(float(lp))
                    if step!=0:
                        belief_results[str(idx)]["diff_logprob"].append(lp - belief_results[str(idx)]["logprob"][-2]) #diff with the previous step
                    if calculate_entropy:
                        belief_results[str(idx)]['entropy'].append(float(entropy_secrets[idx_active]))
                    if return_vocab:
                        belief_results[str(idx)]['vocab'].append(vocab_logprob_secrets[idx_active].tolist())
                    idx_active += 1 

        #for sanity 
        for idx in range(len(self.history)):
            chat_length = len(belief_results[str(idx)]['messages'])
            #odd (missing last user response)
            if chat_length % 2 != 0:
                lp_scores = len(belief_results[str(idx)]['logprob'])
                diff_steps = (chat_length // 2) 
                assert lp_scores == diff_steps, f'lp socres are {lp_scores} but diff_steps are {diff_steps}, idx {idx}, secret {gts[idx]}'

        return belief_results


    def belief_confidence(self,raw_prompt: List[dict], gts: np.ndarray,  game_status: np.ndarray, calculate_entropy:bool=False, return_vocab:bool=False) -> dict:
        """
        Estimate the confidence for the given ground truths.
        """
        
        self.history = [copy.deepcopy(d) for d in raw_prompt.tolist()]
        belief_results = {}
        for idx, gt_secret in enumerate(gts):
            belief_results[str(idx)] = {
                "ground_truth": gt_secret,
                "messages": self.history[idx],
                "confidence": [],
            }
            if game_status is not None:
                belief_results[str(idx)]["game_status"] = game_status[idx]

        #longest trajectory length
        max_length = max([len(hist) for hist in self.history])
        if max_length % 2 == 0:
            max_steps = max_length // 2 
        else:
            max_steps = max_length // 2 
            max_steps += 1
        end = 0
        active = [1 for i in range(len(self.history))]
        for step in tqdm(range(max_steps), desc="Belief Estimation Steps"):
            end += 4
            messages = []
            gt_active = []
            for idx, (hist, gt) in enumerate(zip(self.history, gts)):
                #get out the path trajectory
                if len(hist) > end:
                    # select the last history side
                    current_hist = copy.deepcopy(hist[:end])
                    #adjust the last user response 
                    confidence_reponse = current_hist[-1]["content"] + '. ' + f'Based on the conversation, what do you think is the probability that the answer is "{gt}"? Please answer between 0 and 1.'
                    current_hist[-1] = {"role": "user", "content": confidence_reponse}
                    #append the messages
                    messages.append(current_hist)
                    print(f'[DEBUG] messages: {messages[-1]}')
                    gt_active.append(gt)
                else:
                    active[idx] = 0 

            if not messages:
                break
            
            #now format as token ids 
            rollings = self._format_input(messages, generation_prompt=True)
            rollings.batch = self.tensor_fn.cut_to_effective_len(rollings.batch,keys=['input_ids', 'attention_mask', 'position_ids'],cut_left=True)
 
            # DEBUGGING
            if self.config.debug:
                print(f'[DEBUG] Game round {step+1} \n')
                input_str = self.tokenizer.decode(
                                rollings.batch['input_ids'][0], 
                                skip_special_tokens=False)
                print('[DEBUG] Current input prompt to actor \n', input_str)
                
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(rollings, self.actor_rollout_wg.world_size) #bsz, prompt_len
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            meta_info = test_output_gen_batch_padded.meta_info 
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            responses_ids, responses_str = self._postprocess_responses(test_output_gen_batch.batch['responses'])

            #extract the any number from the response_str
            idx_active = 0
            for idx, active_status in enumerate(active):
                if active_status:
                    response = responses_str[idx_active]
                    print(f'[DEBUG] response: {response}')
                    if response:
                        confidence = float(re.search(r'\d+\.\d+', response).group())
                    print(f'[DEBUG] confidence: {confidence}')
                    belief_results[str(idx)]['confidence'].append(confidence)
                    idx_active += 1

        return belief_results

    def belief_accuracy(self,raw_prompt: List[dict], gts: np.ndarray,  game_status: np.ndarray, calculate_entropy:bool=False, return_vocab:bool=False) -> dict:
        pass
