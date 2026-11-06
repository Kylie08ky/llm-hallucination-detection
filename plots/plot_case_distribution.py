import matplotlib.pyplot as plt
"""
Script: plot_case_distribution.py

Purpose:
    Visualize the distribution of evaluation cases across
    different stages of the evaluation pipeline.

    The figure shows:
        - Gate Fail cases
        - Post-Gate cases
        - Correct cases
        - Incorrect (hallucination) cases

Inputs:
    - Hard-coded counts from the evaluation dataset.

Outputs:
    - Distribution_of_Cases_Across_Evaluation_Stages.png

Run:
    python plots/plot_case_distribution.py

Generates:
    - Bar chart showing the distribution of cases across
      evaluation stages and prediction outcomes.

Used in thesis:
    - Document: Ausarbeitung
    - Section: 7.2.1 Gesamtumfang des Testdatensatzes
    - Figure title:
      "Distribution of Cases Across Evaluation Stages"

Notes:
    - Total dataset size: 200 cases
    - Used to illustrate the proportion of gate failures
      and hallucination cases in the evaluation dataset.
"""

total = 200
data = {
    "Gate Fail": 10,
    "Post-Gate": 190,
    "Correct": 90,
    "Incorrect(hallucination)": 110
}

labels = list(data.keys())
values = list(data.values())

plt.figure(figsize=(8, 5))
bars = plt.bar(labels, values)

plt.ylim(0, max(values) * 1.15)

for bar in bars:
    height = bar.get_height()
    percent = height / total * 100
    
    if height < 30:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + 5,
            f"{int(height)} ({percent:.1f}%)",
            ha='center',
            va='bottom',
            fontsize=10
        )
    else:
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height * 0.95,
            f"{int(height)}\n({percent:.1f}%)",
            ha='center',
            va='top',
            fontsize=10,
            color='white'
        )

plt.ylabel("Number of Cases", fontsize=11)
plt.title("Distribution of Cases Across Evaluation Stages", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("Distribution of Cases Across Evaluation Stages.png", dpi=300)
plt.show()