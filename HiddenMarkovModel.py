# Hongyu Zhang
# Hidden Markov Model

import math
import random
import numpy
from collections import *

class HMM:
	"""
	Represent a Hidden Markov Model.
	"""
	def __init__(self, order, initial_distribution, emission_matrix, transition_matrix):
		self.order = order
		self.initial_distribution = initial_distribution
		self.emission_matrix = emission_matrix
		self.transition_matrix = transition_matrix


def read_pos_file(filename):
	"""
	Reads a text file.
	Input: filename
	Returns: The file represented as a list of tuples, where each tuple is of the form (word, POS-tag).
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


def compute_counts(training_data, order):
	"""
	This function computes the counts of the training data, given the order of the HMM.
	:param training_data: a list of (word, POS-tag) pairs.
	:param order: order of the HMM.
	:return: If order is 2, the function returns a tuple containing the number of tokens in training_data,
	a dictionary that contains that contains C(ti,wi), a dictionary that contains C(ti),
	and a dictionary that contains C(ti-1,ti).
	If order is 3, the function returns as the fifth element a dictionary that contains C(ti-2, ti-1, ti),
	in addition to the other four elements.
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



