import csv, json
rows = list(csv.DictReader(open("../data/harmful_behaviors.csv")))
json.dump([{"instruction": r["goal"], "input": ""} for r in rows],
          open("../data/advbench.json", "w"), indent=1)
print(len(rows), "prompts")