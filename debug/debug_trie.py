import torch

def get_token_id_at_time_step(tokens, time_step):
    # Assuming there is only one batch TODO: handle multiple batches
    if time_step < len(tokens[0]):
        return tokens[0][time_step].item()
    else:
        raise ValueError(f"Time step {time_step} is out of range for the given tokens")

if __name__ == "__main__":
    # Your input
    generated_tokens = torch.tensor([[28740, 28734, 28740, 2]])
    detailed_history = [
        [[{'token_id': 28734, 'token': '0', 'raw_logit': 2.0020976080559194e-05},
          {'token_id': 28740, 'token': '1', 'raw_logit': 3.175825986545533e-05}]],
        [[{'token_id': 28734, 'token': '0', 'raw_logit': 0.22081588208675385},
          {'token_id': 28740, 'token': '1', 'raw_logit': 0.17334003746509552}]],
        [[{'token_id': 28734, 'token': '0', 'raw_logit': 0.11201044172048569},
          {'token_id': 28740, 'token': '1', 'raw_logit': 0.1298251450061798}]],
        [[{'token_id': 2, 'token': '</s>', 'raw_logit': 0.7505450248718262}]]
    ]

    root_success_rate = 2.699477035937618e-05
    full_rate = 2.0020976080559194e-05 + 3.175825986545533e-05 * 0.20367281430155926
    success_rate_0 = 2.0020976080559194e-05 / full_rate
    success_rate_1 = 3.175825986545533e-05 * 0.20367281430155926 / full_rate

    print(f"full_rate", full_rate)
    print(f"success_rate_0", success_rate_0)
    print(f"success_rate_1", success_rate_1)

    for time_step in range(4):
        token_id = get_token_id_at_time_step(generated_tokens, time_step)
        print(f"token_id", token_id)

