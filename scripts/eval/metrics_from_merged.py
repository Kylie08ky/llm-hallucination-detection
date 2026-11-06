import argparse
import os
import pandas as pd

def cm_metrics(d):
    tp = int(((d.y_true==1) & (d.y_pred==1)).sum())
    fp = int(((d.y_true==0) & (d.y_pred==1)).sum())
    fn = int(((d.y_true==1) & (d.y_pred==0)).sum())
    tn = int(((d.y_true==0) & (d.y_pred==0)).sum())
    acc = (tp+tn)/max(1,(tp+fp+fn+tn))
    prec = tp/max(1,(tp+fp))
    rec  = tp/max(1,(tp+fn))
    f1 = 0 if (prec+rec)==0 else 2*prec*rec/(prec+rec)
    return tp,fp,fn,tn,acc,prec,rec,f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="Path to merged_cases.csv")
    ap.add_argument("--out_dir", required=True, help="Directory to save metrics outputs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.merged)

    hard = df[df["hard_fail"] == True]
    post = df[df["hard_fail"] == False].copy()

    # y_true / y_pred
    post["y_true"] = (post["status"] != "CORRECT").astype(int)
    post["y_pred"] = (post["blocked"] == True).astype(int)

    tp,fp,fn,tn,acc,prec,rec,f1 = cm_metrics(post)

    print("=== FILE ===", args.merged)
    print(f"Total: {len(df)} | Hard-fail: {len(hard)} | Post-gate: {len(post)}")
    print("=== POST-GATE CONFUSION MATRIX ===")
    print("TP, FP, FN, TN =", tp, fp, fn, tn)
    print("Accuracy =", round(acc,4))
    print("Precision=", round(prec,4))
    print("Recall   =", round(rec,4))
    print("F1       =", round(f1,4))

    fn_path = os.path.join(args.out_dir, "FN_wrong_but_passed.csv")
    fp_path = os.path.join(args.out_dir, "FP_correct_but_blocked.csv")

    post[(post.y_true==1) & (post.y_pred==0)].to_csv(fn_path, index=False)
    post[(post.y_true==0) & (post.y_pred==1)].to_csv(fp_path, index=False)

    print("[OK] wrote:")
    print("-", fn_path)
    print("-", fp_path)

if __name__ == "__main__":
    main()
