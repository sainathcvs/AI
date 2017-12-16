#Spam filter code using Multinomial Term frequency concept
import csv
import sys
import math

#command line parsing
argv = sys.argv
args={}
while len(argv)>0:
	if argv[0][0] == '-':
		if argv[1][0]!='-':
			args[argv[0]]=argv[1]
		else:
			print 'ERROR: Invalid arguments passed.'
			sys.exit(1)
	argv = argv[1:]
if '-f2' in args.keys():
	test_file = args['-f2']
if '-f1' in args.keys():
	train_file= args['-f1']
if '-o' in args.keys():
	output_file = args['-o']

f = open(train_file, "rb")
reader = csv.reader(f, delimiter=' ')
#number of spam and ham samples
spamCount =0
hamCount = 0
#total number of samples
count = 0
#number of spam and ham words in the training set
spWc= 0
hWc = 0
#dictionaries to maintain counts of spam and ham words in training set
spW = {}
hW = {}
#Iterate over all the training samples 
for row in reader:
	#To know how many documents are there
	count+=1
	idxList = range(2, len(row)-1)
	idxList = idxList[::2] 
	#check if sample is spam or ham and populate corresponding variables
	if row[1]=='spam':
		spamCount+=1
		for i in idxList:
			spWc += int(row[i+1])
			if row[i] in spW:
				spW[row[i]]+=int(row[i+1])
			else:
				spW[row[i]]=int(row[i+1])
	else:
		hamCount+=1
		for i in idxList:
			hWc += int(row[i+1])
			if row[i] in hW:
				hW[row[i]]+=int(row[i+1]) 
			else:
				hW[row[i]]=int(row[i+1])
f.close()

#calculating priors of spam and ham(taking log of them)
spamPrior = math.log(float(spamCount)/count)
hamPrior = math.log(float(hamCount)/count)

#Total vocabulary count in training set
totalWFreqCount = spWc+hWc

#testing
f = open(test_file, "rb")
reader = csv.reader(f, delimiter=' ')
acList = []
correctCount = 0
actualcount = 0
csv_writer = open(output_file, 'wb')
spamwriter = csv.writer(csv_writer, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
#Iterate over all the testing samples
for row in reader:
	actualcount+=1
	acList.append(row[1])
	spamLikelihood = 0
	hamLikelihood = 0
	#For every word in the current test sample
	for i in range(2, len(row)-1):
		if i%2==0:
			#calculating the likelihood of the word given the spam class
			if row[i] in spW: 
				x = float(int(spW[row[i]])+1)/(spWc+totalWFreqCount) 
			else:
				#Laplacian Additive smoothing
				x = 1/float(spWc+totalWFreqCount)
				#Enable the following the line for without additive smoothing
				x = 1
			#calculating the likelihood of the word given the ham class
			if row[i] in hW:
				y = float(int(hW[row[i]])+1)/(hWc+totalWFreqCount)
			else:
				#Laplacian Additive smoothing
				y = 1/float(hWc+totalWFreqCount)
				#Enable the following the line for without additive smoothing
				y = 1
			#log likelihood of all the words in the current test sample
			spamLikelihood += math.log(x)
			hamLikelihood += math.log(y)
	#test whether the current sample belongs to spam or ham
	if(spamLikelihood+spamPrior > hamLikelihood+hamPrior):
		if row[1]=='spam':
			spamwriter.writerow([row[0]]+[row[1]]) 
			correctCount+=1
	else:
		if row[1]=='ham':
			spamwriter.writerow([row[0]]+[row[1]])
			correctCount+=1
f.close()

print "OUTPUT: Accuracy is ", (float(correctCount)/actualcount)*100