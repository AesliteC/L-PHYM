from __future__ import annotations

import argparse
from pathlib import Path

import torch

import MoConVQCore.Utils.pytorch_utils as ptu
from Script.stage1.text_encoding import encode_text_to_feature
from Script.stage1.train_text_gpt import build_text_gpt_model, gpt_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-bvh", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    from Script.moconvq_builder import build_agent

    agent, env = build_agent(gpu=args.gpu)
    ptu.init_gpu(True, gpu_id=args.gpu)
    agent.simple_load("moconvq_base.data", strict=True)
    agent.eval()

    model = build_text_gpt_model(gpt_config(), device=ptu.device, base_data_path="moconvq_base.data")
    state = torch.load(args.checkpoint, map_location="cpu")
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    bert_feature, bert_mask = encode_text_to_feature(args.text)
    bert_feature = torch.as_tensor(bert_feature, dtype=torch.float32, device=ptu.device)
    bert_mask = torch.as_tensor(bert_mask, dtype=torch.bool, device=ptu.device)
    clip_feature = torch.zeros((1, 512), device=ptu.device)
    cur_embedding, _ = model.sample(clip_feature, bert_feature, bert_mask)
    dconv = agent.posterior.decoder.decode_dynamic(cur_embedding)

    import VclSimuBackend

    CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
    saver = CharacterToBVH(agent.env.sim_character, 120)
    saver.bvh_hierarchy_no_root()

    observation, info = agent.env.reset(0)

    for i in range(dconv.shape[1]):
        obs = observation["observation"]
        action, info = agent.act_tracking(
            obs_history=[obs.reshape(1, 323)],
            target_latent=dconv[:, i],
        )
        action = ptu.to_numpy(action).flatten()
        for j in range(6):
            saver.append_no_root_to_buffer()
            if j == 0:
                step_generator = agent.env.step_core(action, using_yield=True)
            info = next(step_generator)

        try:
            info_ = next(step_generator)
        except StopIteration as e:
            info_ = e.value
        new_observation, rwd, done, info = info_
        observation = new_observation

    Path(args.output_bvh).parent.mkdir(parents=True, exist_ok=True)
    saver.to_file(args.output_bvh)


if __name__ == "__main__":
    main()
