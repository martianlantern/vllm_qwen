import os
os.environ['VLLM_USE_V1']='0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch

from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info

if __name__ == '__main__':

    MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    # MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

    llm = LLM(
            model=MODEL_PATH, trust_remote_code=True, gpu_memory_utilization=0.95,
            limit_mm_per_prompt={'image': 0, 'video': 0, 'audio': 1},
            max_num_seqs=4,
            max_model_len=16384,
            dtype=torch.float16
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=16384,
    )

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe the audio content."},
                {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/cookbook/sound1.wav"},
            ], 
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        "mm_processor_kwargs": {
            "use_audio_in_video": True,
        },
    }

    # if images is not None:
    #     inputs['multi_modal_data']['image'] = images
    # if videos is not None:
    #     inputs['multi_modal_data']['video'] = videos
    if audios is not None:
        inputs['multi_modal_data']['audio'] = audios

    outputs = llm.generate([inputs], sampling_params=sampling_params)

# metrics=RequestMetrics(
#     arrival_time=1763216391.670061,
#     last_token_time=1763216398.9073935,
#     first_scheduled_time=1763216398.399984,
#     first_token_time=1763216398.6743326,
#     time_in_queue=6.729922771453857,
#     finished_time=1763216398.9077063,
#     scheduler_time=0.003161802887916565,
#     model_forward_time=None,
#     model_execute_time=None,
#     spec_token_acceptance_counts=[0]
# )
    num_tokens = len(outputs[0].outputs[0].token_ids)
    metrics = outputs[0].metrics
    time_in_queue = metrics.time_in_queue
    time_to_first_token = metrics.first_token_time - metrics.first_scheduled_time
    e2e_latency = metrics.finished_time - metrics.first_scheduled_time
    time_per_output_token = e2e_latency / num_tokens
    inter_token_latency = (metrics.last_token_time - metrics.first_token_time) / (num_tokens - 1)

    print(f"Time to first token: {time_to_first_token*1000} ms")
    print(f"E2E latency: {e2e_latency*1000} ms")
    print(f"Time per output token: {time_per_output_token*1000} ms")
    print(f"Inter-token latency: {inter_token_latency*1000} ms")
    print(f"Time in queue: {time_in_queue*1000} ms")

    print(outputs[0].outputs[0].text)

