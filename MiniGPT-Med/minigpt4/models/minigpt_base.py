import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.base_model import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList

from minigpt4.datasets.data_utils import move_to_cuda
from minigpt4.conversation.conversation import StoppingCriteriaSub

class MiniGPTBase(BaseModel):
    """
    Base class for MiniGPT-4 and MiniGPT-v2
    """

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        llama_model="",
        max_txt_len=32,
        max_context_len=3800,
        prompt_template="",
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        lora_r=0,  # lora_r means lora is not used
        lora_target_modules=["q_proj", "v_proj"],
        lora_alpha=16,
        lora_dropout=0.05,
    ):
        super().__init__()

        self.llama_model, self.llama_tokenizer = self.init_llm(
            llama_model_path=llama_model,
            low_resource=low_resource,
            low_res_device=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision, freeze_vit
        )

        self.max_txt_len = max_txt_len
        self.max_context_len = max_context_len
        self.end_sym = end_sym

        self.prompt_template = prompt_template
        self.prompt_list = []

    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def get_context_emb(self, prompt, img_list):
        device = img_list[0].device
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i==0).to(device).input_ids # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]

        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs

    def prompt_wrap(self, img_embeds, atts_img, prompts, lengths=None):
        if prompts is None or len(prompts) == 0:
            # prompts is not provided, just return the original image embedding
            return img_embeds, atts_img
        elif img_embeds is None:
            # prompt is provided but there is no image embedding. return the prompt embedding in right padding
            self.llama_tokenizer.padding_side = "right"
            prompt_tokens = self.llama_tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                add_special_tokens=False
            ).to(self.device)
            prompt_embeds = self.embed_tokens(prompt_tokens.input_ids)
            atts_prompt = prompt_tokens.attention_mask
            return prompt_embeds, atts_prompt
        else:
            # return the multi-modal embedding in right padding
            emb_lists = []
            
            ############## handle the case of ref and study ###################
            if isinstance(img_embeds, tuple) or isinstance(img_embeds, list): 
                device = img_embeds[0].device
                if isinstance(prompts, str):
                    prompts = [prompts] * len(img_embeds[0])
                ref_embeds, study_embeds = img_embeds
                for idx, (each_ref_embed, each_study_embed, each_prompt) in enumerate(zip(ref_embeds, study_embeds, prompts)):
                    pn = each_ref_embed.shape[-2]
                    stacked_img_embed = torch.cat([each_ref_embed, each_study_embed], dim=-2) 
                    p_segs = each_prompt.split('<ImageHere>')
                    interleave_emb = []
                    for idx, seg in enumerate(p_segs[:-1]):
                        p_tokens = self.llama_tokenizer(
                            seg, return_tensors="pt", add_special_tokens=False).to(device)
                        p_embed = self.embed_tokens(p_tokens.input_ids)
                        interleave_emb.append(torch.cat([p_embed, stacked_img_embed[None][:, idx * pn:(idx + 1) * pn]], dim=1))
                    wrapped_emb = torch.cat(interleave_emb, dim=1)
                    p_tokens = self.llama_tokenizer(
                        p_segs[-1], return_tensors="pt", add_special_tokens=False).to(device)
                    p_embed = self.embed_tokens(p_tokens.input_ids)
                    wrapped_emb = torch.cat([wrapped_emb, p_embed], dim=1)
                    emb_lists.append(wrapped_emb)
            else:
                device = img_embeds.device
                if isinstance(prompts, str):
                    prompts = [prompts] * len(img_embeds)
                for idx, (each_img_embed, each_prompt) in enumerate(zip(img_embeds, prompts)):
                    pn = each_img_embed.shape[-2]
                    if lengths is not None:
                        each_img_embed = each_img_embed.reshape(-1, each_img_embed.shape[-1])
                        each_img_embed = each_img_embed[:lengths[idx] * pn]
                    p_segs = each_prompt.split('<ImageHere>')
                    interleave_emb = []
                    for idx, seg in enumerate(p_segs[:-1]):
                        p_tokens = self.llama_tokenizer(
                            seg, return_tensors="pt", add_special_tokens=False).to(device)
                        p_embed = self.embed_tokens(p_tokens.input_ids)
                        interleave_emb.append(torch.cat([p_embed, each_img_embed[None][:, idx * pn:(idx + 1) * pn]], dim=1))
                    wrapped_emb = torch.cat(interleave_emb, dim=1)
                    p_tokens = self.llama_tokenizer(
                        p_segs[-1], return_tensors="pt", add_special_tokens=False).to(device)
                    p_embed = self.embed_tokens(p_tokens.input_ids)
                    wrapped_emb = torch.cat([wrapped_emb, p_embed], dim=1)
                    emb_lists.append(wrapped_emb)

            emb_lens = [emb.shape[1] for emb in emb_lists]
            pad_emb = self.embed_tokens(torch.tensor(self.llama_tokenizer.pad_token_id, device=device))

            max_length = max(emb_lens) if max(emb_lens) < self.max_context_len else self.max_context_len
            wrapped_embs = pad_emb.expand(len(emb_lens), max_length, -1).clone()
            wrapped_atts = torch.zeros([len(emb_lens), max_length], dtype=torch.int, device=device)
            
            for i, emb in enumerate(emb_lists):
                length = emb_lens[i] if emb_lens[i] < self.max_context_len else self.max_context_len
                wrapped_embs[i, :length] = emb[:, :length]
                wrapped_atts[i, :length] = 1
            return wrapped_embs, wrapped_atts

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        """
        Concatenate the batched input embedding and batched output embedding together.
        Both the input and the output embedding should be right padded.
        """
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    def tokenize_conversation(self, conv_q, conv_a):
        """concatenate conversation and make sure the model is only trained to regress the answer"""

        to_regress_token_ids_list = []
        targets_list = []

        batch_size = len(conv_q)
        for batch_idx in range(batch_size):
            questions, answers = conv_q[batch_idx], conv_a[batch_idx]
            questions = [self.llama_tokenizer(self.llama_tokenizer.bos_token + q,
                                              return_tensors="pt",
                                              add_special_tokens=False).to(self.device) for q in questions[1:]]  # the first question is handled in the prompt wrap function, skip it
            answers = [self.llama_tokenizer(a + self.end_sym,
                                            return_tensors="pt",
                                            add_special_tokens=False).to(self.device) for a in answers]
            cur_id = []
            cur_target = []
            for i in range(len(questions)):
                cur_id.append(answers[i].input_ids)
                cur_target.append(answers[i].input_ids)
                cur_id.append(questions[i].input_ids)
                cur_target.append(torch.ones_like(questions[i].input_ids) * -100)

            cur_id.append(answers[-1].input_ids)
            cur_target.append(answers[-1].input_ids)

            cur_id = torch.cat(cur_id, dim=1)
            cur_target = torch.cat(cur_target, dim=1)
            to_regress_token_ids_list.append(cur_id)
            targets_list.append(cur_target)

        max_len = min(max([target.shape[1] for target in targets_list]), self.max_txt_len)
        to_regress_token_ids = torch.ones([batch_size, max_len],
                                          dtype=cur_id.dtype, device=self.device) * self.llama_tokenizer.pad_token_id
        targets = torch.ones([batch_size, max_len],
                                          dtype=cur_id.dtype, device=self.device) * -100
        for batch_idx in range(batch_size):
            cur_len = to_regress_token_ids_list[batch_idx].shape[1]
            to_regress_token_ids[batch_idx, :cur_len] = to_regress_token_ids_list[batch_idx][0, :max_len]
            targets[batch_idx, :cur_len] = targets_list[batch_idx][0, :max_len]

        to_regress_token_attn = (to_regress_token_ids != self.llama_tokenizer.pad_token_id).to(torch.int)

        return to_regress_token_ids, to_regress_token_attn, targets

    def preparing_embedding(self, samples):
        ### prepare input tokens
        if 'image' in samples:
            img_embeds, img_atts = self.encode_img(samples["image"])
        elif 'ref_feature' in samples and 'study_feature' in samples:
            ref_feats, study_feats = samples['ref_feature'], samples['study_feature']
            ref_embeds, ref_atts = self.proj_features(ref_feats) # [bsz, 305, 4096]
            study_embeds, study_atts = self.proj_features(study_feats) # [bsz, 305, 4096]
            img_embeds = (ref_embeds, study_embeds)
            img_atts = (ref_atts, study_atts)
        else:
            img_embeds = img_atts = None

        if 'conv_q' in samples:
            # handeling conversation datasets
            conv_q, conv_a = samples['conv_q'], samples['conv_a']

            connect_sym = samples['connect_sym'][0]
            conv_q = [q.split(connect_sym)for q in conv_q]
            conv_a = [a.split(connect_sym) for a in conv_a]

            conv_q = [[self.prompt_template.format(item) for item in items] for items in conv_q]

            cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, [q[0] for q in conv_q])
            regress_token_ids, regress_atts, part_targets = self.tokenize_conversation(conv_q, conv_a)

        else:
            if "instruction_input" in samples:
                instruction = samples["instruction_input"]
            elif self.prompt_list:
                instruction = random.choice(self.prompt_list)
            else:
                instruction = None

            if hasattr(self, 'chat_template') and self.chat_template:
                instruction = [self.prompt_template.format(instruct) for instruct in instruction]

            if 'length' in samples: # not applicable for diff-vqa task
                # the input is a image train (like videos)
                bsz, pn, hs = img_embeds.shape
                img_embeds = img_embeds.reshape(len(samples['image']), -1, pn, hs)
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction, samples['length'])
            else:
                cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction)

            ### prepare target tokens
            self.llama_tokenizer.padding_side = "right"
            text = [t + self.end_sym for t in samples["answer"]]

            regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(self.device)

            regress_token_ids = regress_tokens.input_ids
            regress_atts = regress_tokens.attention_mask
            part_targets = regress_token_ids.masked_fill(
                regress_token_ids == self.llama_tokenizer.pad_token_id, -100
            )

        regress_embeds = self.embed_tokens(regress_token_ids)

        return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets

    def forward(self, samples, reduction='mean'):
        # prepare the embedding to condition and the embedding to regress
        cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets = \
            self.preparing_embedding(samples)

        # concat the embedding to condition and the embedding to regress
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)

        # get bos token embedding
        bos = torch.ones_like(part_targets[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = cond_atts[:, :1]

        # add bos token at the begining
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)

        # ensemble the final targets
        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)

        for i, target in enumerate(part_targets):
            targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target  # plus 1 for bos

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                reduction=reduction
            )
        loss = outputs.loss

        return {"loss": loss}

    def embed_tokens(self, token_ids):
        if hasattr(self.llama_model.base_model, 'model'): ## lora wrapped model
            embeds = self.llama_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.llama_model.base_model.embed_tokens(token_ids)
        return embeds

    @torch.no_grad()
    def generate(
        self,
        images,
        texts,
        num_beams=1,
        max_new_tokens=20,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1,
        length_penalty=1,
        temperature=1,
        do_sample=False,
        stop_words_ids=[2],
    ):
        '''
            function for generate test use
        '''

        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
            stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])])
        
        if isinstance(images, tuple) or isinstance(images, list):
            ref_features, study_features = images[0], images[1]
            ref_embeds, _ = self.proj_features(move_to_cuda(ref_features)) # [bs, pn, hs] 
            study_embeds, _ = self.proj_features(move_to_cuda(study_features)) # [bs, pn, hs]
            image_lists = [[ref_embed[None], study_embed[None]] for ref_embed, study_embed in zip(ref_embeds, study_embeds)]
        else:
            # img_embeds, atts_img = self.encode_img(images.to(self.device))
            img_embeds = torch.randn(images.size(0), 256, 4096).to(self.device) # for debug only
            image_lists = [[image_emb[None]] for image_emb in img_embeds]

        batch_embs = [self.get_context_emb(text, img_list) for text, img_list in zip(texts, image_lists)]

        batch_size = len(batch_embs)
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype
        device = batch_embs[0].device

        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1

        with self.maybe_autocast():
            outputs = self.llama_model.generate(
                inputs_embeds=embs,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                temperature=temperature,
                do_sample=do_sample,
                min_length=min_length,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                # stopping_criteria=stopping_criteria,
            )

        # with self.maybe_autocast():
        #     outputs = self.llama_model.generate(
        #         inputs_embeds=embs,
        #         attention_mask=attn_mask,
        #         max_new_tokens=max_new_tokens,
        #         num_beams=num_beams,
        #         do_sample=do_sample,
        #         # stopping_criteria=stopping_criteria,
        #     )
        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.llama_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('</s>')[0]  # remove the stop sign </s>
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)

        return answers

    @torch.no_grad()
    def multi_select(self, images, texts, answers, num_cand=None):
        all_losses = []
        for answer in answers:
            choice_samples = {
                'image': images,
                'instruction_input': texts,
                'answer': answer
            }
            loss = self.forward(choice_samples, reduction='none')['loss'].reshape(-1, 1)
            all_losses.append(loss)
            torch.cuda.empty_cache()
        all_losses = torch.cat(all_losses, dim=-1)
        if num_cand is not None:
            for i in range(all_losses.shape[0]):
                all_losses[i, num_cand[i]:] = 9999
        output_class_ranks = torch.argsort(all_losses, dim=-1)
        return output_class_ranks.tolist()
