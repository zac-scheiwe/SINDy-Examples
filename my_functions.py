import numpy as np

def reformat(feature_names):
    axis_nums = [1, 2]
    axis_names = ["x", "y"]
    new_names = [feature_names.pop(0)]
    for num, name in zip(axis_nums, axis_names):
        feature_names = [c.replace(str(num), name) for c in feature_names]
    new_names += feature_names
    return(new_names)

def print_result(model, solution_name, optimizer_name):
    print("Feature library:")
    features = model.feature_library.get_feature_names(solution_name)
    print(reformat(features))
    print(f"{optimizer_name} model:")
    model.print()

def lognormal_noisify(u, sd):
    norm_sigma = np.sqrt(np.log(1+sd**2))
    norm_mu = -norm_sigma**2/2
    return(np.multiply(np.random.lognormal(norm_mu, norm_sigma, u.shape), u))

# Test:
# n = 100
# abc = np.arange(1, 1+n**2).reshape([n,n])
# efg = noisify(abc, 0.1)
# ratios = np.divide(efg, abc)
# np.mean(ratios), np.std(ratios)

def noisify(u, sd):
    return(u+np.random.normal(0, sd, u.shape))
