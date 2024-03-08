import time
import tqdm

if __name__ == "__main__":
    log_file_path = '/nobackup2/yf/mila/GD/log/debug2.log'
    input_grammar = '/nobackup2/yf/mila/GD/examples/grammars/string_start_w_1_all_0.ebnf'
    iter = 3
    result = 1
    with open(log_file_path, 'a') as log:
        log.write(f"input_grammar: {input_grammar}")
        for i in tqdm(range(iter), desc="Running Inference"):
            log.write(f"result: {result}")
            log.flush()
            # print(f'start logging...')
            res = result[0].split(".")[2]
            # print(f"res: {res}")
            if res in output:
                output[res] += 1
            else:
                output['other'] += 1

            faithful[res] = faithful.get(res, 0) + 1 # collect all the outputs instead of classifying to others
            if i % 10 == 0:
                log.write(f"Iteration: {i+1}")
                log.flush()
                log.write(f"Output: {output}")
                log.flush()
                log.write(f"Faithful: {faithful}")
                log.flush()
        end_time = time.time()
        elapsed_time = end_time - start_time
        log.write(f"Elapsed time: {elapsed_time} seconds")
        log.write(f"model_id: {args.model_id}")
        log.write(f"repetition_penalty: {args.repetition_penalty}")
        # print(f"num_beams: {args.num_beams}")
        log.write(f"temperature: {args.temperature}")
        log.write(f"top_p: {args.top_p}")
        log.write(f"max_new_tokens: {args.max_new_tokens}")
        log.write(f"output: {output}")
        log.write(f"faithful: {faithful}")
        log.write(f"ideal: {ideal}")
    return output, faithful, ideal, elapsed_time