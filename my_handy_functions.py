def reformat(feature_names):
    axis_nums = [1, 2]
    axis_names = ["x", "y"]
    new_names = [feature_names.pop(0)]
    for num, name in zip(axis_nums, axis_names):
        feature_names = [c.replace(str(num), name) for c in feature_names]
    new_names += feature_names
    return(new_names)

def print_result(model, solution_name):
    print("Feature library:")
    features = model.feature_library.get_feature_names(solution_name)
    print(reformat(features))
    print("STLSQ model:")
    model.print()

