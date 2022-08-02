def synthesize_result(log_file: str):
    log_file = open(log_file, "r")
    result = {}
    max_indice = []
    results_lines = log_file.readlines()
    log_file.close()
    for line in results_lines:
        if "Max value to replace" in line:
            max_value = line.split()[4]
            index = results_lines.index(line)
            if index not in max_indice:
                max_indice.append(index)
            if max_value not in result.keys():
                result[float(max_value)] = []
    for index, key in enumerate(result.keys()):
        if index < len(max_indice) - 1:
            result[key] = results_lines[max_indice[index]:max_indice[index + 1]]
        else:
            result[key] = results_lines[max_indice[index]:]
    for key in result.keys():
        exception_input = set()
        for i, line in enumerate(result[key]):
            if "The input" in line:
                exception_input.add(float(line.split()[3][:-1]))
        result[key] = exception_input
    exceptions_per_max_value = []
    result_file = open("Xscope_result.txt", "w")
    for key in result.keys():
        result_file.write("With max value set to {}, here are the list of input that caused exception: \n".format(key))
        for input in result[key]:
            result_file.write("\t" + str(input) + "\n")
        total_exception = len(result[key])
        exceptions_per_max_value.append(total_exception)
        result_file.write("Total input found: {}".format(len(result[key])))
        result_file.write("\n")
    result_file.close()

