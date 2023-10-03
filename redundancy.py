import csv

input_file = 'indian_liver_patient.csv'
output_file = 'ilpd.csv'

with open(input_file, 'r') as csv_file, open(output_file, 'w', newline='') as output:
    reader = csv.reader(csv_file)
    writer = csv.writer(output)
    rows_seen = set()
    for row in reader:
        if tuple(row) not in rows_seen:
            writer.writerow(row)
            rows_seen.add(tuple(row))
