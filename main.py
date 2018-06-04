#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import affinegap
from os import listdir
import json
import heapq

import torch.nn as nn
import torch
from torch.autograd import Variable
from string import punctuation

import model

NBCONTEXTE = 1
hidden_size = 6
input_size = NBCONTEXTE*2 + 1
learning_rate = 0.001
SEUIL = 0.5
seuilDist = True

num_epoch = 1


def format(tourParole = True):
    saisons = []
    nomsSaisons = []
    locuteurSaisons = []
    full = []
    
    allLocS = []
    for nEp in range(1, 11):
        episodes = []
        nomsEpisodes = []
        locuteurEpisodes = []
        allLoc = []
        
        for f in listdir("/people/galmant/premierModele/spacy.anotations.new/tbbt.season" + str(nEp) + "/"):
            ep = open("/people/galmant/premierModele/spacy.anotations.new/tbbt.season" + str(nEp) + "/" + f, 'r')
            
            phrases = []
            npps = []
            locuteurs = []
            
            
            
            phrase = ""
            locPrec = ""
            npp = []
            prenomTrouve = 0
            prevSp = ""
            
            for line in ep:
                sp = line.split(' ')
                if (sp[1] != locPrec and tourParole) or (not tourParole and any(p in sp[1] for p in punctuation)):
                    if prenomTrouve:
                        npp.append(prevSp)
                        prenomTrouve = 0
                    phrases.append(phrase)
                    npps.append(npp)
                    npp = []
                    phrase = ""
                    locPrec = sp[1]
                    locuteurs.append(sp[1])
                    
                if sp[3] != "X":
                    #npp.append(sp[0])
                    if prenomTrouve:
                        if prevSp != sp[0]:
                            npp.append(prevSp + "_" + sp[0])
                        else:
                            npp.append(prevSp)
                        prenomTrouve = 0
                    else:
                        prenomTrouve = 1
                else:
                    if prenomTrouve:
                        npp.append(prevSp)
                        prenomTrouve = 0
                phrase += sp[0] + " "
                prevSp = sp[0]
                
                
                
                if sp[1] not in full:
                    full.append(sp[1])
                    allLoc.append(sp[1])
                
            phrases.append(phrase)
            npps.append(npp)
            locuteurs.append(sp[1])
            
            episodes.append(phrases)
            nomsEpisodes.append(npps)
            locuteurEpisodes.append(locuteurs)
            
        saisons.append(episodes)
        nomsSaisons.append(nomsEpisodes)
        locuteurSaisons.append(locuteurEpisodes)
        allLocS.append(allLoc)
        
    #print(allLocS[0])
    return saisons, nomsSaisons, locuteurSaisons, allLocS
    
"""def getCandidats(nSaison):
    candidats = []
    
    for f in listdir("/people/galmant/premierModele/spacy.anotations.new/tbbt.season" + str(nSaison) + "/"):
        ep = open("/people/galmant/premierModele/spacy.anotations.new/tbbt.season" + str(nSaison) + "/" + f, 'r')
        for line in ep:
            sp = line.split(' ')
            if sp[1] not in candidats:
                candidats.append(sp[1])
                
    return candidats"""
    
def ddd():
    for loc in locuteurSaisons:
        for lo in loc:
            for l in lo:
                print(l)
       
def getCandidats():
    with open("out.json", "r") as f:
        data = json.load(f)
        
    data = data["0898266"]
    per = ""
    perSplit = []
    
    
    noms = []
    
    saisons = {}
    saisons["2006"] = (1, 1)
    saisons["2007"] = (1, 1)
    saisons["2008"] = (1, 2)
    saisons["2009"] = (2, 3)
    saisons["2010"] = (3, 4)
    saisons["2011"] = (4, 5)
    saisons["2012"] = (5, 6)
    saisons["2013"] = (6, 7)
    saisons["2014"] = (7, 8)
    saisons["2015"] = (8, 9)
    saisons["2016"] = (9, 10)
    saisons["2017"] = (10, 11)
    saisons["2018"] = (11, 11)
    
    candidats = [[],[],[],[],[],[],[],[],[],[],[]]
    
    for pers in data:
        per = pers[2]
        perSplit = per.split(" ")
        stop = 0
        nom = ""
        
        for w in perSplit:
            if not stop:
                if not w.isdigit() and "/" not in w and "(" not in w and "#" not in w:
                    nom += w + "_"
                else:
                    stop = 1
        nom = nom[:-1].lower()
        noms.append(nom)
        
        annees = perSplit[-1]
        if "-" not in annees:
            prem, sec = saisons[annees]
            if nom not in candidats[prem-1]:
                candidats[prem-1].append(nom)
            if nom not in candidats[sec-1]:
                candidats[sec-1].append(nom)
        else:
            anneesSplit = annees.split("-")
            annee1 = int(anneesSplit[0])
            annee2 = int(anneesSplit[1])
            
            for i in range(annee1, annee2 + 1):
                prem, sec = saisons[str(i)]
                if nom not in candidats[prem-1]:
                    candidats[prem-1].append(nom)
                if nom not in candidats[sec-1]:
                    candidats[sec-1].append(nom)
            
                
    return candidats
        #print(annees)
        
    #print(noms)

def getVectFeature(candidat, indexPhrase, noms, seuil = True):
    dist = []
    #vec = np.ones(1 + 2*NBCONTEXTE)
    vec = torch.ones(1, 1 + 2*NBCONTEXTE)
    
    i = 0
    for d in range(indexPhrase - NBCONTEXTE, indexPhrase + NBCONTEXTE +1):
        if d >= 0 and d < len(noms):
            for nom in noms[d]:
                #dist.append(affinegap.normalizedAffineGapDistance(nom, candidat, matchWeight = 0, mismatchWeight = 1))
                dist.append(damerau_levenshtein_distance(nom, candidat))
                if not seuil:
                    if dist:
                        vec[0][i] = min(dist)
                        dist = []
                else:
                    if dist:
                        if min(dist) < SEUIL:
                            vec[0][i] = 0
                        dist = []
        i += 1
        
    return vec

#candidats = getCandidats()

def getTrainSet():
    saisons = range(4, 11)
    for saison in saisons:
        pass
    
def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1
 
    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                           d[(i-1,j)] + 1, # deletion
                           d[(i,j-1)] + 1, # insertion
                           d[(i-1,j-1)] + cost, # substitution
                          )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition
                
    size = lenstr1 if lenstr1 > lenstr2 else lenstr2
 
    return d[lenstr1-1,lenstr2-1]/size

    
#print(candidats[0])
def normalizeCandidat():
    saison = 9
    
    saisons, nomsSaisons, locuteurSaisons, allLoc = format()
    candidats = getCandidats()
    c = 0
    dic = {}
    
    
    for loc in allLoc[saison]:
        print(c)
        dist = []
        print("Qui correspond à " + loc.replace("unknown_", "") + " ?\n")
        for cand in candidats[saison]:
            #dist.append(affinegap.normalizedAffineGapDistance(loc, cand, matchWeight = 0, mismatchWeight = 1, gapWeight = 10, spaceWeight = 5))
            dist.append(damerau_levenshtein_distance(loc.replace("unknown_", ""), cand))
        distM = heapq.nsmallest(10, dist)
        i = 0
        for d in distM:
            print(str(i) + ". " + candidats[saison][dist.index(d)])
            i += 1
        print(str(i) + ". Autre")
        print(str(i+1) + ". Stop")
    
        choix = input("Votre choix : ")
        if int(choix) == i+1:
            print(dic)
            with open("saison10.json", 'w') as f:
                json.dump(dic, f)
        elif int(choix) == i:
            pass
        else:
            dic[loc] = candidats[saison][dist.index(distM[int(choix)])]
        c += 1
    with open("saison10.json", 'w') as f:
        json.dump(dic, f)
    
        
#normalizeCandidat()

    
saisons, nomsSaisons, locuteurSaisons, _ = format()

def getInput(numSaison):
    
    saison = saisons[numSaison]
    nomsSaison = nomsSaisons[numSaison]
    locuteursSaison = locuteurSaisons[numSaison]
    
    x = []
    y = []
    
    with open("format.json", "r") as f:
        data = json.load(f)
    
    candidats = getCandidats()[numSaison]
    print(str(len(candidats)) + " candidats")
    
    for i in range(0, len(saison)):
        episode = saison[i]
        nomsEpisode = nomsSaison[i]
        locuteursEpisode = locuteursSaison[i]
        
        for (m, loc) in enumerate(locuteursEpisode):
            if loc in data:
                locuteursEpisode[m] = data[loc]
                
        for (j, phrase) in enumerate(episode):
            vectors = []
            yt = []
            for cand in candidats:
                vectors.append(getVectFeature(cand, j, nomsEpisode, seuilDist))
                if cand == locuteursEpisode[j]:
                    yt.append(1.0)
                else:
                    yt.append(0.0)
            x.append(np.array(vectors))
            y.append(np.array(yt))
                    
    return x, y

def getTrain():
    x = []
    y = []
    #for i in range(4,10):
    for i in range(4,5):
        xt, yt = getInput(i)
        for it in xt:
            x.append(it)
            
        for it in yt:
            y.append(it)
    return x, y
        
"""def getTrain():
    x, y = getInput(4)
    for i in range(5,10):
        
        xt, yt = getInput(i)
        print(str(len(x)) + ", " + str(len(xt)))
        x = np.concatenate((x, xt))
        y = np.concatenate((y, yt))
        
    return x, y
"""
def getEval():
    x = []
    y = []
    #for i in range(2,4):
    for i in range(2,3):
        xt, yt = getInput(i)
        for it in xt:
            x.append(it)
            
        for it in yt:
            y.append(it)
    return x, y

cpt = 0

for saison in saisons:
    for ep in saison:
        cpt += len(ep)
        
print (cpt)
        

x, y = getTrain()

bm = model.BinaryModel(input_size, hidden_size)

criterion = nn.BCELoss()  
optimizer = torch.optim.Adam(bm.parameters(), lr=learning_rate)


nbT = 700
x = x[:nbT]

for epoch in range(num_epoch):
    for j, xt in enumerate(x):
        for i, inp in enumerate(xt):
            res = y[j][i]
            #print(inp)
            inp = Variable(inp)
           # inp = Variable(inpu)
            res = [res]
    
            res = Variable(torch.FloatTensor(res))
            
            optimizer.zero_grad() 
            
            out = bm(inp)
            
            
            loss = criterion(out, res)
            loss.backward()
            optimizer.step()
            
            if (j+1) % 10000 == 0 and i == 0:
                
                print("loss : ")
                print(loss.data[0])
                print("vec : ")
                print(inp)
                print("\n\n")
                
    
print("Eval\n")
x, y = getEval()
#print(x[0])
correct = 0
total = 0
choix1 = 0

printout = 0

x = x[:700]

for (j, tp) in enumerate(x):
    
    out = []
    for v in tp:
        
        v = Variable(v)
        o = bm(v)
        out.append(o.data[0][0])
        if v.data[0][0] <0.0:
            """print("v : ")
            print(v[0][0])
            print("o : ")
            print(o)"""
            printout = 1
    choix = out.index(max(out))
    if printout:
        print("out : ")
        print(out)
        print("res : ")
        print(max(out))
        printout = 0
        
    if y[j][0] != 1:
        if y[j][choix] == 1:
            correct += 1
        total += 1
    """if choix == 0:
        choix1 += 1
    if y[j][choix] == 1:
        correct += 1
    total += 1"""
    
#print(100 * choix1 / total)
print("Seuil : " + str(SEUIL) + ", nbT : " + str(nbT) + ", seuilDist : " + str(seuilDist))
print(100 * correct / total)

    
    
"""_, noms, _, _ = format()

for i in range(0, 300):
    print(getVectFeature("sheldon_cooper", i, noms[0][0]))"""
        

#print(affinegap.normalizedAffineGapDistance('sheldon_cooper', 'penny', matchWeight = 0, mismatchWeight = 1))

#print(nump.array(phrases).shape)
#print(nump.array(npps).shape)
    
"""for phrase in phrases:
    print(phrase + "\n")"""
    
"""for nom in npps:
    for n in nom:
        print(n)
    print("\n")"""