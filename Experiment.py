# Hongyu Zhang
# Experiment

import math
import pylab
import types
import numpy
from collections import *

class HMM:
	"""
	Simple class to represent a Hidden Markov Model.
	"""

	def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
		self.order = order
		self.initial_distribution = initial_distribution
		self.emission_matrix = emission_matrix
		self.transition_matrix = transition_matrix


def read_pos_file(filename):
	"""
	Parses an input tagged text file.
	Input:
	filename --- the file to parse
	Returns:
	The file represented as a list of tuples, where each tuple
	is of the form (word, POS-tag).
	A list of unique words found in the file.
	A list of unique POS tags found in the file.
	"""
	file_representation = []
	unique_words = set()
	unique_tags = set()
	f = open(str(filename), "r")
	for line in f:
		if len(line) < 2 or len(line.split("/")) != 2:
			continue
		word = line.split("/")[0].replace(" ", "").replace("\t", "").strip()
		tag = line.split("/")[1].replace(" ", "").replace("\t", "").strip()
		file_representation.append((word, tag))
		unique_words.add(word)
		unique_tags.add(tag)
	f.close()
	return file_representation, unique_words, unique_tags


def _dict2lists(data):
	"""
	Convert a dictionary into a list of keys and values, sorted by
	key.

	Arguments:
	data -- dictionary

	Returns:
	A tuple of two lists: the first is the keys, the second is the values
	"""
	xvals = data.keys()
	xvals.sort()
	yvals = []
	for x in xvals:
		yvals.append(data[x])
	return xvals, yvals


def _plot_dict_line(d, label=None):
	"""
	Plot data in the dictionary d on the current plot as a line.

	Arguments:
	d     -- dictionary
	label -- optional legend label

	Returns:
	None
	"""
	xvals, yvals = _dict2lists(d)
	if label:
		pylab.plot(xvals, yvals, label=label)
	else:
		pylab.plot(xvals, yvals)


def plot_lines(data, title, xlabel, ylabel, labels=None, filename=None):
	"""
	Plot a line graph with the provided data.

	Arguments:
	data     -- a list of dictionaries, each of which will be plotted
				as a line with the keys on the x axis and the values on
				the y axis.
	title    -- title label for the plot
	xlabel   -- x axis label for the plot
	ylabel   -- y axis label for the plot
	labels   -- optional list of strings that will be used for a legend
				this list must correspond to the data list
	filename -- optional name of file to which plot will be
				saved (in png format)

	Returns:
	None
	"""
	### Check that the data is a list
	if not isinstance(data, types.ListType):
		msg = "data must be a list, not {0}".format(type(data).__name__)
		raise TypeError(msg)

	### Create a new figure
	fig = pylab.figure()

	### Plot the data
	if labels:
		mylabels = labels[:]
		for i in range(len(data)-len(labels)):
			mylabels.append("")
		for d, l in zip(data, mylabels):
			_plot_dict_line(d, l)
		# Add legend
		pylab.legend(loc='best')
		gca = pylab.gca()
		legend = gca.get_legend()
		pylab.setp(legend.get_texts(), fontsize='medium')
	else:
		for d in data:
			_plot_dict_line(d)

	### Set the lower y limit to 0 or the lowest number in the values
	mins = [min(l.values()) for l in data]
	ymin = min(0, min(mins))
	pylab.ylim(ymin=ymin)

	### Label the plot
	pylab.title(title)
	pylab.xlabel(xlabel)
	pylab.ylabel(ylabel)

	### Draw grid lines
	pylab.grid(True)

	### Show the plot
	fig.show()

	### Save to file
	if filename:
		pylab.savefig(filename)


def compute_counts(training_data, order):
	"""
	This function computes the counts of the training data, given the order of the HMM.
	:param training_data: a list of (word, POS-tag) pairs.
	:param order: order of the HMM.
	:return: If order is 2, the function returns a tuple containing the number of tokens in training_data, a dictionary that contains that contains C(ti,wi), a dictionary that contains C(ti), and a dictionary that contains C(ti-1,ti).
			 If order is 3, the function returns as the fifth element a dictionary that contains C(ti-2, ti-1, ti), in addition to the other four elements.
	"""
	dict1 = defaultdict(lambda: defaultdict(int))
	dict2 = defaultdict(int)
	dict3 = defaultdict(lambda: defaultdict(int))
	words_list = [training_data[idx][0] for idx in range(0, len(training_data))]
	tags_list = [training_data[idx][1] for idx in range(0, len(training_data))]

	if order == 2:
		num_token = len(training_data)
		# Compute C(ti)
		for tag_i in tags_list:
			dict2[tag_i] += 1
		# Compute C(ti, wi)
		for idx1 in range(0, num_token):
			dict1[tags_list[idx1]][words_list[idx1]] += 1
		# Compute C(ti-1, ti)
		if len(training_data) >= 2:
			for idx2 in range(0, len(training_data)-1):
				dict3[tags_list[idx2]][tags_list[idx2+1]] += 1

		return num_token, dict1, dict2, dict3

	elif order == 3:
		dict4 = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
		num_token = len(training_data)
		# Compute C(ti)
		for tag_i in tags_list:
			dict2[tag_i] += 1
		# Compute C(ti, wi)
		for idx1 in range(0, num_token):
			dict1[tags_list[idx1]][words_list[idx1]] += 1
		# Compute C(ti-1, ti)
		if len(training_data) >= 2:
			for idx2 in range(0, len(training_data) - 1):
				dict3[tags_list[idx2]][tags_list[idx2 + 1]] += 1
		# Compute C(ti-2, ti-1, ti)
		if len(training_data) >= 3:
			for idx3 in range(0, len(training_data) - 2):
				dict4[tags_list[idx3]][tags_list[idx3+1]][tags_list[idx3+2]] += 1

		return num_token, dict1, dict2, dict3, dict4

	else:
		raise Exception('The order must be 2 or 3.')


def compute_initial_distribution(training_data, order):
	"""
	This function computes the initial distribution given order of HMM and training data.
	:param training_data: a list of (word, POS-tag) pairs.
	:param order: order of the HMM.
	:return: returns a dictionary that contains pi1 if order equals 2, and pi2 if order equals 3.
	"""
	length = len(training_data)
	tags_list = [training_data[idx][1] for idx in range(0, length)]

	if order == 2:
		count_dict = defaultdict(int)
		pi_dict = defaultdict(float)
		# Count the tags that appear in the first of the sentence.
		count_dict[tags_list[0]] += 1
		for idx1 in range(0, length-1):
			if tags_list[idx1] == ".":
				if tags_list[idx1+1] != ".":
					count_dict[tags_list[idx1+1]] += 1
		total_counts = sum(count_dict.values())
		# Calculate the probability.
		for tag_i in count_dict.keys():
			pi_dict[tag_i] += float(count_dict[tag_i]) / total_counts

		return pi_dict

	elif order == 3:
		count_dict = defaultdict(lambda: defaultdict(int))
		pi_dict = defaultdict(lambda: defaultdict(float))
		# Count the tags that appear in the first of the sentence.
		count_dict[tags_list[0]][tags_list[1]] += 1
		for idx1 in range(0, length-2):
			if tags_list[idx1] == ".":
				if tags_list[idx1+1] != ".":
					if tags_list[idx1+2] != ".":
						count_dict[tags_list[idx1+1]][tags_list[idx1+2]] += 1
		total_counts = 0
		for tag1 in count_dict.keys():
			total_counts += sum(count_dict[tag1].values())
		# Calculate the probability.
		for tag_i in count_dict.keys():
			for tag_j in count_dict[tag_i].keys():
				pi_dict[tag_i][tag_j] += float(count_dict[tag_i][tag_j]) / total_counts

		return pi_dict

	else:
		raise Exception('The order must be 2 or 3.')


def compute_emission_probabilities(unique_words, unique_tags, W, C):
	"""
	This function computes the emission matrix.
	:param unique_words: set returned by read_pos_file
	:param unique_tags: set returned by read_pos_file
	:param W: C(ti,wi)
	:param C: C(ti)
	:return: a dictionary of emission matrix.
	"""
	emission = defaultdict(lambda: defaultdict(float))
	for tag in unique_tags:
		for word in unique_words:
			if W[tag][word] != 0:
				emission[tag][word] += float(W[tag][word]) / C[tag]

	return emission


def compute_lambdas(unique_tags, num_tokens, C1, C2, C3, order):
	"""
	This function implements algorithm Compute Lambda.
	:param unique_tags: the set returned by read_pos_file.
	:param num_tokens: total number of tokens in the training corpus.
	:param C1: C(ti)
	:param C2: C(ti-1, ti)
	:param C3: C(ti-2, ti-1, ti)
	:param order: order of the HMM.
	:return: a list that contains lambda0, lambda1, lambda2.
	"""
	# order 3
	if order == 3:
		lambda_list = [0.0, 0.0, 0.0]
		for t1 in unique_tags:
			for t2 in unique_tags:
				for t3 in unique_tags:
					if C3[t1][t2][t3] > 0:
						# If denominator is not 0.
						if num_tokens != 0:
							a0 = float((C1[t3] - 1)) / num_tokens
						else:
							a0 = 0
						# If denominator is not 0.
						if (C1[t2] - 1) != 0:
							a1 = float((C2[t2][t3] - 1)) / (C1[t2] - 1)
						else:
							a1 = 0
						# If denominator is not 0.
						if (C2[t1][t2] - 1) != 0:
							a2 = float((C3[t1][t2][t3]) - 1) / (C2[t1][t2] - 1)
						else:
							a2 = 0

						i = numpy.argmax([a0, a1, a2])
						lambda_list[i] += float(C3[t1][t2][t3])

		lambda0 = float(lambda_list[0]) / sum(lambda_list)
		lambda1 = float(lambda_list[1]) / sum(lambda_list)
		lambda2 = float(lambda_list[2]) / sum(lambda_list)

		return [lambda0, lambda1, lambda2]

	# order 2
	elif order == 2:
		lambda_list = [0.0, 0.0]
		for t1 in unique_tags:
			for t2 in unique_tags:
				if C2[t1][t2] > 0:
					# If denominator is not 0.
					if num_tokens != 0:
						a0 = float((C1[t2] - 1)) / num_tokens
					else:
						a0 = 0
					# If denominator is not 0.
					if (C1[t1] - 1) != 0:
						a1 = float((C2[t1][t2] - 1)) / (C1[t1] - 1)
					else:
						a1 = 0

					i = numpy.argmax([a0, a1])
					lambda_list[i] += float(C2[t1][t2])

		lambda0 = float(lambda_list[0]) / sum(lambda_list)
		lambda1 = float(lambda_list[1]) / sum(lambda_list)

		return [lambda0, lambda1, 0.0]

	else:
		raise Exception('The order must be 2 or 3.')


def compute_transition_matrix(unique_tags, num_tokens, c1, c2, c3, order, lambdas_list):
	"""
	This helper function computes the transition matrix.
	:param unique_tags: the set returned by read_pos_file.
	:param num_tokens: total number of tokens in the training corpus.
	:param c1: C(ti)
	:param c2: C(ti-1, ti)
	:param c3: C(ti-2, ti-1, ti)
	:param order: order of the HMM.
	:param lambdas_list: lambdas.
	:return: transition matrix.
	"""
	if order == 2:
		# Compute transition matrix.
		transition_matrix = defaultdict(lambda: defaultdict(float))
		for t1 in unique_tags:
			for t2 in unique_tags:
				if c1[t1] != 0:
					transition_matrix[t1][t2] += (lambdas_list[1] * float(c2[t1][t2]) / c1[t1]
												  + lambdas_list[0] * float(c1[t2]) / num_tokens)
	else:
		# Compute transition matrix.
		transition_matrix = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
		for t1 in unique_tags:
			for t2 in unique_tags:
				for t3 in unique_tags:
					if c2[t1][t2] != 0 and c1[t2] != 0:
						transition_matrix[t1][t2][t3] += (lambdas_list[2] * float(c3[t1][t2][t3]) / c2[t1][t2]
														  + lambdas_list[1] * float(c2[t2][t3]) / c1[t2]
														  + lambdas_list[0] * float(c1[t3]) / num_tokens)

	return transition_matrix


def build_hmm(training_data, unique_tags, unique_words, order, use_smoothing):
	"""
	This function returns a fully trained HMM.
	:param training_data: a list of (word, POS-tag) pairs.
	:param unique_tags: the set returned by read_pos_file.
	:param unique_words: the set returned by read_pos_file.
	:param order: order of the HMM.
	:param use_smoothing: a Boolean parameter.
	:return: This function returns a fully trained HMM.
	"""
	# Compute initial distribution.
	initial_dist = compute_initial_distribution(training_data, order)
	num_token = compute_counts(training_data, order)[0]
	w = compute_counts(training_data, order)[1]
	c1 = compute_counts(training_data, order)[2]
	c2 = compute_counts(training_data, order)[3]
	# Compute emission matrix.
	emission_matrix = compute_emission_probabilities(unique_words, unique_tags, w, c1)
	# order 2
	if order == 2:
		# Compute transition matrix.
		lambdas_list = [0.0, 1.0, 0.0]
		if use_smoothing:
			lambdas_list = compute_lambdas(unique_tags, num_token, c1, c2, {}, 2)
		transition_matrix = compute_transition_matrix(unique_tags, num_token, c1, c2, {}, 2, lambdas_list)
		# Build HMM.
		hmm = HMM(2, initial_dist, emission_matrix, transition_matrix)
		return hmm
	# order 3
	elif order == 3:
		c3 = compute_counts(training_data, 3)[4]
		# Compute transition matrix.
		lambdas_list = [0.0, 0.0, 1.0]
		if use_smoothing:
			lambdas_list = compute_lambdas(unique_tags, num_token, c1, c2, c3, 3)
		transition_matrix = compute_transition_matrix(unique_tags, num_token, c1, c2, c3, 3, lambdas_list)

		# Build hmm.
		hmm = HMM(3, initial_dist, emission_matrix, transition_matrix)
		return hmm

	else:
		raise Exception('The order must be 2 or 3.')


def bigram_viterbi(hmm, sentence):
	"""
	This function implements the Viterbi algorithm for the bigram model.
	:param hmm: an HMM.
	:param sentence: sequence of observations.
	:return: the sentence tagged in the form of a list of (word, tag) pairs.
	"""
	# Get the initial distribution, transition matrix, emission matrix and states.
	emi_matrix = hmm.emission_matrix
	init_dist = hmm.initial_distribution
	transi_matrix = hmm.transition_matrix
	states = transi_matrix.keys()

	# Initialize v, bp, z, and length.
	v = defaultdict(lambda: defaultdict(float))
	bp = defaultdict(lambda: defaultdict(str))
	length = len(sentence)
	z = [0 for _ in range(0, length)]

	# Compute first column.
	for state in states:
		if init_dist[state] == 0.0 or emi_matrix[state][sentence[0]] == 0.0:
			v[state][0] = -float("inf")
		else:
			v[state][0] += math.log(init_dist[state]) + math.log(emi_matrix[state][sentence[0]])

	# Compute v and bp.
	for i in range(1, length):
		for state1 in states:
			list1 = []
			for state2 in states:
				if v[state2][i-1] == -float("inf") or transi_matrix[state2][state1] == 0.0:
					list1.append(-float("inf"))
				else:
					list1.append(v[state2][i-1]+math.log(transi_matrix[state2][state1]))

			if emi_matrix[state1][sentence[i]] == 0.0:
				v[state1][i] += -float("inf")
			else:
				v[state1][i] += math.log(emi_matrix[state1][sentence[i]]) + max(list1)
			bp[state1][i] += states[numpy.argmax(list1)]

	# Compute Z.
	list2 = []
	for state_i in states:
		if v[state_i][length-1] == -float("inf"):
			list2.append(-float("inf"))
		else:
			list2.append(v[state_i][length-1])
	z[length-1] = states[numpy.argmax(list2)]

	for j in range(length-2, -1, -1):
		z[j] = bp[z[j+1]][j+1]

	result = []
	for idx in range(0, length):
		result.append((sentence[idx], z[idx]))

	return result


def trigram_viterbi(hmm, sentence):
	"""
	This function implements the Viterbi algorithm for the trigram model.
	:param hmm: an HMM.
	:param sentence: sequence of observations.
	:return: the sentence tagged in the form of a list of (word, tag) pairs.
	"""
	# Get the initial distribution, transition matrix, emission matrix and states.
	emi_matrix = hmm.emission_matrix
	init_dist = hmm.initial_distribution
	transi_matrix = hmm.transition_matrix
	states = transi_matrix.keys()

	# Initialize v, bp, z, and L.
	v = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
	bp = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
	length = len(sentence)
	z = [0 for _ in range(0, length)]

	# Compute first column.
	for tag1 in states:
		for tag2 in states:
			if init_dist[tag1][tag2] == 0.0 or emi_matrix[tag1][sentence[0]] == 0.0 or emi_matrix[tag2][sentence[1]] == 0.0:
				v[tag1][tag2][1] += -float("inf")
			else:
				v[tag1][tag2][1] += (math.log(init_dist[tag1][tag2])
										+ math.log(emi_matrix[tag1][sentence[0]])
										+ math.log(emi_matrix[tag2][sentence[1]]))
	# Compute v and bp.
	for i in range(2, length):
		for m in states:
			for n in transi_matrix[m].keys():
				list1 = []
				for l in states:
					if v[l][m][i-1] == -float("inf") or transi_matrix[l][m][n] == 0.0:
						list1.append(-float("inf"))
					else:
						list1.append(v[l][m][i-1] + math.log(transi_matrix[l][m][n]))

				if emi_matrix[n][sentence[i]] == 0.0:
					v[m][n][i] += -float("inf")
				else:
					v[m][n][i] += math.log(emi_matrix[n][sentence[i]]) + max(list1)
				bp[m][n][i] += states[numpy.argmax(list1)]

	# Compute Z.
	max_value = -float("inf")
	s1 = None
	s2 = None
	for tag1 in states:
		for tag2 in states:
			if v[tag1][tag2][length-1] == -float("inf"):
				value = -float("inf")
			else:
				value = v[tag1][tag2][length-1]
			if value >= max_value:
				max_value = value
				s1 = tag1
				s2 = tag2
	z[length-2] = s1
	z[length-1] = s2
	for j in range(length-3, -1, -1):
		z[j] = bp[z[j+1]][z[j+2]][j+2]

	result = []
	for idx in range(0, length):
		result.append((sentence[idx], z[idx]))

	return result


def split_sentences(filename):
	"""
	This function splits paragraph into sentences.
	:param filename: filename.
	:return: a list that contains all the sentences.
	"""
	text_file = open(filename, "r")
	sentence = text_file.read().split(" ")
	result = []
	index = 0
	for idx in range(0, len(sentence)):
		if sentence[idx] == ".":
			result.append(sentence[index:idx + 1])
			index = idx + 1

	return result


def update_hmm(hmm, testing_file, unique_words):
	"""
	This function handles unknown words.
	:param hmm: an HMM.
	:param unique_words: the set of unique words in the training file.
	:param testing_file: the testing file.
	:return: update the emission matrix.
	"""
	# Update and normalize emission matrix.
	emission_matrix = hmm.emission_matrix

	text_file = open(testing_file, "r")
	words = text_file.read().split(" ")
	for idx in range(0, len(words)):
		if words[idx] == "":
			words.pop(idx)
	text_file.close()

	new_word = []
	for word in words:
		if word not in unique_words:
			new_word.append(word)

	if new_word != []:
		# Add the probability.
		for i in emission_matrix:
			for j in emission_matrix[i]:
				if emission_matrix[i][j] != 0.0:
					emission_matrix[i][j] += 0.00001
			for word in new_word:
				emission_matrix[i][word] += 0.00001

		# Normalize the probability.
		normalize_prob = defaultdict(float)
		for i in emission_matrix:
			normalize_prob[i] += sum(emission_matrix[i].values())
			for j in emission_matrix[i]:
				emission_matrix[i][j] = 1.0 * emission_matrix[i][j] / normalize_prob[i]


def compute_accuracy(prediction_list, actual):
	"""
	This function computes the accuracy of our prediction.
	:param prediction_list: predicted result.
	:param actual: the actual result.
	:return: the accuracy.
	"""
	total_list = []
	count = 0.0
	for item in prediction_list:
		for element in item:
			total_list.append(element)
	length = len(total_list)
	for idx in range(0, length):
		if total_list[idx][1] == actual[idx][1]:
			count += 1.0

	accuracy = count/length
	return accuracy


def experiment(order, use_smoothing, training_file, testing_file, actual_file):
	"""
	This function builds 7 HMMs and train them and obtain their accuracy values.
	:param order: order of HMM.
	:param use_smoothing: a Boolean parameter.
	:param training_file: training filename.
	:param testing_file: testing filename.
	:param actual_file: the correct (word, tag) pairs filename.
	:return: a list that contains the accuracy values.
	"""
	accuracy_dist = {}
	# Split the paragraph into sentences.
	sentences = split_sentences(testing_file)
	training_data = read_pos_file(training_file)[0]
	actual_result = read_pos_file(actual_file)[0]

	for percent in [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00]:
		threshold = int(round(len(training_data) * percent))
		training_data1 = training_data[0:threshold]
		unique_tags = set([])
		unique_words = set([])
		for i in range(threshold):
			unique_tags.add(training_data1[i][1])
			unique_words.add(training_data1[i][0])

		# Build and update hmm.
		hmm = build_hmm(training_data1, unique_tags, unique_words, order, use_smoothing)
		update_hmm(hmm, testing_file, unique_words)

		prediction_list = []
		if order == 2:
			for sentence in sentences:
				prediction_list.append(bigram_viterbi(hmm, sentence))
		elif order == 3:
			for sentence in sentences:
				prediction_list.append(trigram_viterbi(hmm, sentence))
		else:
			raise Exception('The order must be 2 or 3.')
		# Calculate the accuracy.
		accuracy_dist[percent] = compute_accuracy(prediction_list, actual_result)

	return accuracy_dist


# Experiment 1
# print experiment(2, False, "training.txt", "testdata_untagged.txt", "testdata_tagged.txt")
# {0.01: 0.7063655030800822, 0.05: 0.8275154004106776, 0.1: 0.9014373716632443, 0.25: 0.9373716632443532,
# 0.5: 0.9507186858316222, 0.75: 0.9640657084188912, 1.0: 0.9691991786447639}

# Experiment 2
# print experiment(3, False, "training.txt", "testdata_untagged.txt", "testdata_tagged.txt")
# {0.01: 0.35831622176591377, 0.05: 0.7002053388090349, 0.1: 0.7915811088295688, 0.25: 0.8726899383983573,
# 0.5: 0.9301848049281314, 0.75: 0.9363449691991786, 1.0: 0.9404517453798767}

# Experiment 3
# print experiment(2, True, "training.txt", "testdata_untagged.txt", "testdata_tagged.txt")
# {0.01: 0.7751540041067762, 0.05: 0.8809034907597536, 0.1: 0.9065708418891171, 0.25: 0.9332648870636551,
# 0.5: 0.9486652977412731, 0.75: 0.9640657084188912, 1.0: 0.9681724845995893}

# Experiment 4
# print experiment(3, True, "training.txt", "testdata_untagged.txt", "testdata_tagged.txt")
# {0.01: 0.6878850102669405, 0.05: 0.811088295687885, 0.1: 0.9055441478439425, 0.25: 0.9435318275154004,
# 0.5: 0.9589322381930184, 0.75: 0.9681724845995893, 1.0: 0.973305954825462}

data1 = experiment(2, False, "training.txt", "testdata_untagged.txt", "testdata_tagged.txt")
data2 = experiment(3, False, "training.txt", "testdata_untagged.txt", "testdata_tagged.txt")
data3 = experiment(2, True, "training.txt", "testdata_untagged.txt", "testdata_tagged.txt")
data4 = experiment(3, True, "training.txt", "testdata_untagged.txt", "testdata_tagged.txt")
# Draw the line graph.
plot_lines([data1, data2, data3, data4], "Part of Speech Tag Accuracy", "Proportion of Training Data", "Accuracy Values",
           ["Experiment 1: Bigram without smoothing", "Experiment 2: Trigram without smoothing",
            "Experiment 3: Bigram with smoothing", "Experiment 4: Trigram with smoothing"])
