def reformat_array(feature_names):
    axis_nums = [1, 2]
    axis_names = ["x", "y"]
    new_names = [feature_names.pop(0)]
    for num, name in zip(axis_nums, axis_names):
        feature_names = [c.replace(str(num), name) for c in feature_names]
    new_names += feature_names
    return(new_names)

def reformat_string(s):
    axis_nums = [1, 2]
    axis_names = ["x", "y"]
    for num, name in zip(axis_nums, axis_names):
        # print(num, name)
        s.replace(str(num), name)
    return(s)

from io import StringIO 
import sys

class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio    # free up some memory
        sys.stdout = self._stdout

def print_result(model, solution_name):
    print("Feature library:")
    features = model.feature_library.get_feature_names(solution_name)
    print(reformat_array(features))
    print("STLSQ model:")
    with Capturing() as output:
        model.print()
    print(output)
    # print(reformat_string(output[0]))

