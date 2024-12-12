import csv

# File Path
input_file = "H:/Repository/scitail/SciTailV1.1/tsv_format/scitail_1.0_test.tsv"  # SciTail file path
output_file = "./scitail_test_prompts.txt"

def process_scitail_tsv(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        reader = csv.reader(infile, delimiter="\t")
        for row in reader:
            if len(row) != 3:
                continue
            
            premise, hypothesis, label = row
            # Prompt 
            prompt = (
                f"Premise: {premise}\n"
                f"Hypothesis: {hypothesis}\n"
                # f"Is the hypothesis entailed by the premise or is it neutral?\n"
                # f"Answer:"
            )
            # Write into file
            outfile.write(prompt + "\n\n")
    print(f"Finish! File saved: {output_file}")

# Run
process_scitail_tsv(input_file, output_file)
