def get_llm_output(pkl_file, category_file, dataset_file):
    with open(dataset_file, "r") as file:
        dataset = file.readlines()

    reviews = [x.split("####")[0] for x in dataset]

    raw_true_outputs = [eval(x.split("####")[1]) for x in dataset]
    true_outputs = []
    for output in raw_true_outputs:
        true_outputs.append([tuple(quint) for quint in output])

    with open(pkl_file, "rb") as file:
        llm_outputs = pickle.load(file)

    with open(category_file, "rb") as file:
        categories = pickle.load(file)

    pred_outputs = []
    for i in range(len(llm_outputs)):
        llm_output = llm_outputs[i]
        category = categories[i]

        pred_output = []
        for j in range(len(llm_output)):
            if llm_output[j] == 1:
                pred_output.append(category[j])

        pred_outputs.append(pred_output)

    return reviews, pred_outputs, true_outputs
