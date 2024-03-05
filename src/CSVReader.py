import csv


class CSVReader:

    @staticmethod
    def read_csv(file, types, head=False):

        with open(file, mode='r') as train:
            # Create a CSV reader object
            data = csv.reader(train)
            if not head:
                next(data)

            data_with_typification = []
            for row in data:
                row = list(cast(val.strip()) if val else cast() for cast, val in zip(types, row))
                data_with_typification.append(row)

            return data_with_typification

