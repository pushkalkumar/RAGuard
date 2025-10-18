import json, random

# load_split function --> reads a jsonl file and returns a list of dictionaries
def load_split(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f] 
    

#sample batch function --> Function for generating training batches that have negative values to train the retriever
# Takes in a dataset, batch size, and number of negatives per query
def sample_batch(dataset, batch_size=16, num_negatives=1):
    random.shuffle(dataset)
    # iterate over the data in batches
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        for item in batch:
            negativesamples = []
            while len(negativesamples) < num_negatives:
                neg = random.choice(dataset)
                if neg["doc_id"] != item["doc_id"]:
                    negativesamples.append(neg["doc_text"])
            # Create the triples (query, positive doc, negative docs)
            yield item["query"], item["doc_text"], negativesamples

# Sanity check function --> checks if the dataset is loaded correctly
def sanity_check(train, dev, tests):
    queryids = set() #track all the seen query ids 
    docids = set() #track all the seen document ids
    for split, name in zip([train, dev, tests], ["train", "dev", "test"]):
        for item in split:
            assert item["query_id"] not in queryids, f"Duplicate query id {item['query_id']} in {name} set"
            queryids.add(item["query_id"])
            assert item["doc_id"] not in docids, f"Duplicate doc id {item['doc_id']} in {name} set"
            docids.add(item["doc_id"])
            # ensure that the documents all have reasonable lengths
            assert 1 < len(item["query"]) < 512
            assert 1 < len(item["doc_text"]) < 2048

#Function to generate the training triples
# Each triple contains queries, gold documents, and a negative document
#take the split data and out_path arguments which determines where to save the triples
def generate_triples(split, out_path):
    with open(out_path, 'w') as f:
        for item in split:
            negative = random.choice([i for i in split if i["gold_doc"] != item["gold_doc"]])
            triple = {
                "query": item["query"],
                "gold_doc": item["gold_doc"],
                "neg_doc": negative["gold_doc"],
            }
            # write each of the triples as single lines in the JSONL file
            f.write(json.dumps(triple) + "\n")