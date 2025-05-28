import os
import json

def save_results_to_json(
    args,
    candidates,
    refs,
    human_scores,
    dual_task_parse_results,
    sub_sentences_parse_dict,
    parse_dict,
    spice_score_original,
    spice_score_sub_sentences,
    spice_score_delete,
    spice_score_insert,
    spice_score_combined,
    soft_spice_score_original,
    soft_spice_score_sub_sentences,
    soft_spice_score_delete,
    soft_spice_score_insert,
    soft_spice_score_combined,
    save_path="correlation_results_caparena",
    time_tag=None,
    capture_score_subsentence=None,
    capture_score_delete=None,
    capture_score_insert=None,
    capture_score_combined=None,
):
    output_path = os.path.join(save_path, time_tag)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_file = f"{output_path}/{args.dataset}_correlation_score.json"

    if not os.path.exists(output_file):
        with open(output_file, "w") as f:
            f.write("[]")

    for i in range(len(candidates)):
        cand = candidates[i]
        ref = refs[i]
        human_score = human_scores[i]
        max_len = max(len(ref), len(cand))

        # 提取所有分数
        original_spice_score = spice_score_original[i * max_len : (i + 1) * max_len]
        sub_sentences_spice_score = spice_score_sub_sentences[
            i * max_len : (i + 1) * max_len
        ]
        delete_spice_score = spice_score_delete[i * max_len : (i + 1) * max_len]
        insert_spice_score = spice_score_insert[i * max_len : (i + 1) * max_len]
        combined_spice_score = spice_score_combined[i * max_len : (i + 1) * max_len]
        original_soft_spice_score = soft_spice_score_original[
            i * max_len : (i + 1) * max_len
        ]
        sub_sentences_soft_spice_score = soft_spice_score_sub_sentences[
            i * max_len : (i + 1) * max_len
        ]
        delete_soft_spice_score = soft_spice_score_delete[
            i * max_len : (i + 1) * max_len
        ]
        insert_soft_spice_score = soft_spice_score_insert[
            i * max_len : (i + 1) * max_len
        ]
        combined_soft_spice_score = soft_spice_score_combined[
            i * max_len : (i + 1) * max_len
        ]

        # convert np.float to float
        original_spice_score = [float(x) for x in original_spice_score]
        sub_sentences_spice_score = [float(x) for x in sub_sentences_spice_score]
        delete_spice_score = [float(x) for x in delete_spice_score]
        insert_spice_score = [float(x) for x in insert_spice_score]
        combined_spice_score = [float(x) for x in combined_spice_score]
        original_soft_spice_score = [float(x) for x in original_soft_spice_score]
        sub_sentences_soft_spice_score = [
            float(x) for x in sub_sentences_soft_spice_score
        ]
        delete_soft_spice_score = [float(x) for x in delete_soft_spice_score]
        insert_soft_spice_score = [float(x) for x in insert_soft_spice_score]
        combined_soft_spice_score = [float(x) for x in combined_soft_spice_score]

        # 创建数据字典
        data = {
            "cand": cand,
            "ref": ref,
            "human_score": human_score,
            "original_spice_score": original_spice_score,
            "sub_sentences_spice_score": sub_sentences_spice_score,
            "delete_spice_score": delete_spice_score,
            "insert_spice_score": insert_spice_score,
            "combined_spice_score": combined_spice_score,
            "original_soft_spice_score": original_soft_spice_score,
            "sub_sentences_soft_spice_score": sub_sentences_soft_spice_score,
            "delete_soft_spice_score": delete_soft_spice_score,
            "insert_soft_spice_score": insert_soft_spice_score,
            "combined_soft_spice_score": combined_soft_spice_score,
        }

        # 如果启用了capture，添加相关分数
        if args.capture:
            sub_sent_capture_score = capture_score_subsentence[1][
                i * max_len : (i + 1) * max_len
            ]
            delete_capture_score = capture_score_delete[1][
                i * max_len : (i + 1) * max_len
            ]
            insert_capture_score = capture_score_insert[1][
                i * max_len : (i + 1) * max_len
            ]
            combined_capture_score = capture_score_combined[1][
                i * max_len : (i + 1) * max_len
            ]

            # convert np.float to float
            sub_sent_capture_score = [float(x) for x in sub_sent_capture_score]
            delete_capture_score = [float(x) for x in delete_capture_score]
            insert_capture_score = [float(x) for x in insert_capture_score]
            combined_capture_score = [float(x) for x in combined_capture_score]
            data.update(
                {
                    "sub_sent_capture_score": sub_sent_capture_score,
                    "delete_capture_score": delete_capture_score,
                    "insert_capture_score": insert_capture_score,
                    "combined_capture_score": combined_capture_score,
                }
            )

        with open(output_file, "r") as f:
            existing_data = json.load(f)

        existing_data.append(data)

        with open(output_file, "w") as f:
            json.dump(existing_data, f, indent=4)

    mid_res_path = os.path.join(output_path, "graph_mid_res")
    if not os.path.exists(mid_res_path):
        os.makedirs(mid_res_path)

    # save dual_task_parse_results to json
    dual_task_parse_results_file = (
        f"{mid_res_path}/{args.dataset}_dual_task_parse_results.json"
    )
    with open(dual_task_parse_results_file, "w") as f:
        json.dump(dual_task_parse_results, f, indent=4)
    print(f"Results saved to {dual_task_parse_results_file}")
    # save sub_sentences_parse_dict_object to json
    sub_sentences_parse_dict_file = (
        f"{mid_res_path}/{args.dataset}_sub_sentences_parse_dict_object.json"
    )
    with open(sub_sentences_parse_dict_file, "w") as f:
        json.dump(sub_sentences_parse_dict, f, indent=4)
    print(f"Results saved to {sub_sentences_parse_dict_file}")
    # save parse_dict to json
    parse_dict_file = f"{mid_res_path}/{args.dataset}_parse_dict.json"
    with open(parse_dict_file, "w") as f:
        json.dump(parse_dict, f, indent=4)
    print(f"Results saved to {parse_dict_file}")

    return output_file
