import random

trainingData = []

d = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]
for x in range(30):
    trainingData.append(("number "+str(x), ""))
    trainingData.append((str(x),""))

for x in d:
    trainingData.append((x, ""))
    trainingData.append(("letter "+ x, ""))
    trainingData.append((x + " is a letter", ""))
    trainingData.append(("letter "+ x.upper(), ""))

moretrainingData = [
	[
            ("Hello", "Hi"),
            ("My name is Barack Obama", ""),
            ("I stay in Ney york", ""),
            ("What is your name", "Sheldon Cooper"),
            ("How old are you", "I am 25 years old"),
			
			
            ("What is your name", "My name is Jerry"),
            ("What is your's", "Kathrine McPhee"),
            ("How was your day", "My day was fine, thank you"),
            ("Where are you going to now", "I am going to London"),
            ("Where is that", "It is in the UK"),
            ("What is UK", "UK is the United Kingdom"),
			
			
            ("Hello", "Good Morning"),
            ("Where can I get a bus here", "Downtown"),
            ("Ok", "Thank you"),
            
			("hello", "Hey"),
			("my name is james gordon", ""),
			("What is your name", "my name is carrie underwood"),
			("Where are you staying", "I am stating in Los angeles"),
			
			
			("hello", "hi"),
			("my name is jerry", ""),
			("what is my name", "your name is jerry"),
			("my name is dan williams", ""),
			("what is my name", "your name is dan williams"),
			("my name is martin luther king the third", ""),
			("what is my name", "your name is martin luther king the third"),
			("my name is massie williams jnr", ""),
			("what is my name", "your name is massie williams"),
			("my name is john kennedy", ""),
			("what is my name", "your name is john"),
			("my name is harry potter", ""),
			("what is my name", "your name is harry"),
			
			("janet is my sister", ""),
			("my sister is in lagos", ""),
			
			("lagos is a place", ""),
			("new york is a place", ""),
			("ketu is a place", ""),
			("usa is a place", ""),
			("america is a place", ""),
			("germany is a place", ""),
			("africa is a place", ""),
			("a continent is a place", ""),
			
			("africa is a continent", ""),
			
			("caleb james is in germany", ""),
			("steve jobs is in america", ""),
			("alice scarlett is in africa", ""),
			
			("where is caleb james", "he is in usa"),
			("where is steve jobs", "he is in america"),
			("where is jonathan", "he is in usa"),
			("hello", "Hey"),
			
	],
	[
            tuple([str(x) + " + " + str(xx), str(x+xx)]) for x in range(10) for xx in range(10) ] + [
            tuple([str(x) + " - " + str(xx), str(x-xx)]) for x in range(10) for xx in range(10) ] + [
            tuple([str(x) + " x " + str(xx), str(x*xx)]) for x in range(10) for xx in range(10) ] + [
            tuple([str(x) + " / " + str(xx), str(x/xx)]) for x in range(10) for xx in range(1,20)
	],
	[
            ("count 1 to 5", "") ] + [ tuple([str(a), ""]) for a in range(1, 6) ] + [
            ("count 1 to 7", "") ] + [ tuple([str(a), ""]) for a in range(1, 8) ] + [
            ("count 1 to 9", "") ] + [ tuple([str(a), ""]) for a in range(1, 10) ] + [
            ("count 1 to 10", "") ] + [ tuple([str(b), ""]) for b in range(1, 11) ] + [
            ("count 1 to 13", "") ] + [ tuple([str(b), ""]) for b in range(1, 14) ] + [
            ("count 1 to 15", "") ] + [ tuple(["", str(c)]) for c in range(1, 16) ] + [
            ("count 1 to 17", "") ] + [ tuple(["", str(c)]) for c in range(1, 18) ] + [
            ("count 1 to 20", "") ] + [ tuple(["", str(c)]) for c in range(1, 21) ] + [
            ("count 1 to 23", "") ] + [ tuple(["", str(c)]) for c in range(1, 24) ] + [
            ("count 1 to 29", "") ] + [ tuple(["", str(c)]) for c in range(1, 30)
            
	]
]
for x in moretrainingData: trainingData.extend(x)
trainingData.extend(moretrainingData[-1])
