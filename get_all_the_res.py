import glob
import json
import argparse


def get_cmd():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sort_by", default="model", type=str, help="by which key to sort the results, options: model, recall, mrr, ndcg")
    args = parser.parse_args()
    return args


def main(print_type="csv"):
    sort_by = get_cmd().sort_by
    assert sort_by in ["model", "recall", "mrr", "ndcg"], "only support: model, recall, mrr, ndcg"

    res = {}
    for each_file in glob.glob("./performance/*/*/*/*"):
        x = each_file.split("/")
        dataset = x[2]
        seq_len = int(x[3].split("_")[-1])
        model = x[4]
        setting = x[5]

        if dataset not in res:
            res[dataset] = {}

        if seq_len not in res[dataset]:
            res[dataset][seq_len] = {}

        if model not in res[dataset][seq_len]:
            res[dataset][seq_len][model] = {}

        res_str = []
        for line in open(each_file):
            if "TOP 10: REC_T" in line:
                res_str.append(line)

        if len(res_str) <= 0:
            continue
        y = res_str[-1].strip().split(",")
        epoch = int(y[1].split(" ")[-1])
        recall = float(y[2].split("=")[-1])
        mrr = float(y[3].split("=")[-1])
        ndcg = float(y[4].split("=")[-1])

        res[dataset][seq_len][model][setting] = {"recall": recall, "mrr": mrr, "ndcg": ndcg}

    if print_type == "csv":
        for dataset, x in res.items():
            print("%s:" %(dataset))
            for seq_len, y in sorted(x.items(), key=lambda i: i[0]):
                print("\tseq_len %d:" %(seq_len))
                for model, z in sorted(y.items(), key=lambda i: i[0]):
                    print("\t\t%s:" %(model))
                    sorted_res = None
                    if sort_by == "model":
                        sorted_res = sorted(z.items(), key=lambda i: i[0])
                    else:
                        sorted_res = sorted(z.items(), key=lambda i: i[1][sort_by], reverse=True)

                    print("\t\t\trank:  \tRecall@10  \tMRR@10  \tNDCG@10   \tModelName")
                    for rank, (setting, m) in enumerate(sorted_res):
                        print("\t\t\t%d:\t%f\t%f\t%f\t%s" %(rank, m["recall"], m["mrr"], m["ndcg"], setting))
    else:
        print(json.dumps(res, indent=4))


if __name__ == "__main__":
    main()
