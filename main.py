import cbir
import random

if __name__ == "__main__":

    # create the dataset
    # for the sake of speed, we will do it in a subset
    root = "data/jpg"
    dataset = cbir.Dataset(root)
    subset = dataset.subset[0:10]

    # create the vocabulary tree

    sift = cbir.descriptors.EzSIFT()
    voc = cbir.encoders.VocabularyTree(n_branches=3, depth=3, descriptor=sift)
    voc.learn(subset)
    voc.save()
    # and now create the database
    db = cbir.Database(subset, encoder=voc)

    # let's generate the index
    db.index()

    # and test a retrieval
    query_path = "ukbench00007"
    scores = db.retrieve(query_path)
    db.show_results(query_path,scores)
