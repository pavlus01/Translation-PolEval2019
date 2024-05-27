file1_path = "../Data/TED2020.es-pl.pl"
file2_path = "../Data/TED2020.es-pl.es"
merged_file_path = "../Data/TED2020.es-pl.txt"

with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2, open(merged_file_path, 'w') as merged_file:
    for line1, line2 in zip(file1, file2):
        merged_file.write(line1.strip() + "##" + line2)