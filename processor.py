class Processor:
    def ARM_sessionTracker(self, que, ans, ARM):
        que_list = que.split(" ")
        ans_list = ans.split(" ")
        m, model = [], ""
        aGradients = ARM["aGradients"]
        seperator = ""

        if len(que_list) == 1 and len(ans_list) == 1:
            seperator = ""
            que_modified = " ".join(list(que)).strip()
            ans_modified = " ".join(list(ans)).strip()
            data = que_modified
            dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats = self.generateModelers(data, [que_modified], [ans_modified])
            print(dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats, que_modified, ans_modified, sep="\n")

            # extracting vars from the string map to figure the alterations
            varz = {}
            cfq = commonFormatQue[0][0]
            cfa = commonFormatAns[0][0]
            model = outputFormats[0]

            c = 0 #counter and indexer
            while "[var]" in cfq and "[var]" in cfa:
                varz.setdefault(c, {1:commonFormatQue[0][-1][c], 2:commonFormatAns[0][-1][c]})
                cfa = cfa.replace("[var]", "", 1)
                cfq = cfq.replace("[var]", "", 1)
            
            print("varz = ", varz, ARM["confidence"])
            x, y = {}, {}
            for var in varz:
                que_val, ans_val = (varz[var][i] for i in range(1,3,1))
                que_sessions = self.getMemoryData(que_val, "sessions")
                ans_sessions = self.getMemoryData(ans_val, "sessions")
                
                if que_sessions == None or ans_sessions == None:
                    return {"aGradients":m,"model":model, "seperator":seperator, "confidence":ARM["confidence"]}
                    

                for qsx in que_sessions:
                    for zx in que_sessions[qsx]:
                        z = zx.split(" ")
                        # x.append(qsx+" "+z[0]+" "+z[1])
                        x.setdefault(qsx+" "+z[0], int(z[1]))
                
                for asx in ans_sessions:
                    for zx in ans_sessions[asx]:
                        z = zx.split(" ")
                        # y.append(asx+" "+z[0]+" "+z[1])
                        y.setdefault(asx+" "+z[0], int(z[1]))
            
            # to find the most gradients
            m = {}
            for ix in x:
                if ix in y:
                    m.setdefault(ix, y[ix]-x[ix])

            # if accumulated gradient available, filter
            if len(aGradients) > 1:
                m = {xx:m[xx] for xx in aGradients if xx in m and aGradients[xx] == m[xx]}
            
        # print(m, model)
        # input()
        return {"aGradients":m,"model":model, "seperator":seperator, "confidence":ARM["confidence"]}

    def runInputProcessing(self, data, callback=False):
        que, ans, qna_infl, dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats, output_classes, que_sessions, ans_sessions, in_memory = self.getFeatures(data, callback)
        # for i in range(len(que)):                
        #     self.showProcess("\nque = {}\nans = {}\ndatamap = {}\ncommonFormatQue = {}\ncommonFormatAns = {}\noutputFormat = {}\nsearchFormat = {}\n".format(
        #         que[i], ans[i], dataMap[i], commonFormatQue[i], commonFormatAns[i], outputFormats[i], searchFormats[i]))

        #instantiating variables
        predicted = {}

        # model from accumulated gradients
        model = -1

        #first log influence of repeating [var] values
        outputFormats_infl = {x:outputFormats.count(x) for x in set(outputFormats)}

        for clx in output_classes.copy():
            #check for already defined ans values
            if outputFormats_infl[clx] > 1 and '[var]' not in clx:
                predicted.setdefault(clx, {"class":[clx], "infl": {clx:outputFormats_infl[clx]/len(outputFormats)}})
            cl = clx
            output_class = output_classes[clx]

            temp_que, temp_ans, temp_que_session_index, temp_ans_session_index, temp_qna_infl, temp_dataMap, temp_commonFormatQue, temp_commonFormatAns, temp_searchFormats, temp_outputFormats, temp_search_memory_li = [], [], [], [], [], [], [], [], [], [], []
            for y in output_class:
                temp_que.append(que[y])
                
                temp_ans.append(ans[y])

                temp_que_session_index.append(que_sessions[y])
                
                temp_ans_session_index.append(ans_sessions[y])
                
                temp_qna_infl.append(qna_infl[y])

                #this maps the related que to data
                temp_dataMap.append(dataMap[y])
                
                #this gets the common que in ans   
                temp_commonFormatQue.append(commonFormatQue[y])
                    
                #this gets the common anns in que 
                temp_commonFormatAns.append(commonFormatAns[y])

                #this uses the datamap to turn the individual que to look as that of data
                temp_searchFormats.append(searchFormats[y])
                #this uses the datamap to turn the individual ans to look as that of data
                temp_outputFormats.append(outputFormats[y])

                #check for activation
                temp_search_memory_li.append(in_memory[y])
            
            temp_searchFormat_max = self.getMax([x[0] for x in temp_searchFormats])

            #output format
            temp_outputFormat = cl

            self.showProcess('\nclass: {}'.format(cl))

            self.showProcess("search_memory_li = {}".format(temp_search_memory_li))
            temp_search_memory = max([temp_search_memory_li.count(x) for x in set(temp_search_memory_li)]) == temp_search_memory_li.count(True)
            
            temp_ans_infl = {x:temp_ans.count(x)/len(temp_ans) for x in set(temp_ans)}
            
            for i in range(len(temp_que)):                
                # self.showProcess("\ntemp_que = {}\ntemp_ans = {}\ntemp_datamap = {}\ntemp_commonFormatQue = {}\ntemp_commonFormatAns = {}\ntemp_outputFormat = {}\ntemp_searchFormat = {}\ntemp_que_session_index = {}\ntemp_ans_session_index = {}\ntemp_search_memory = {}\n".format(
                #     temp_que[i], temp_ans[i], temp_dataMap[i], temp_commonFormatQue[i], temp_commonFormatAns[i], temp_outputFormats[i], temp_searchFormats[i], temp_que_session_index[i], temp_ans_session_index[i], temp_search_memory_li[i]))

                if "[var]" == cl:
                    temp_ans_infl_mean = sum(temp_ans_infl.values())/len(temp_ans_infl)
                    temp_ans_infl_treshold = {x:temp_ans_infl[x] - temp_ans_infl_mean for x in temp_ans_infl if temp_ans_infl[x] >= temp_ans_infl_mean}
                    temp_ans_infl_treshold_sum = sum(temp_ans_infl_treshold.values())

                    if temp_ans_infl_treshold_sum > 0:
                        temp_ans_infl_treshold =  {x:temp_ans_infl_treshold[x]/temp_ans_infl_treshold_sum for x in temp_ans_infl_treshold}
                    
                    if temp_ans_infl_treshold_sum == 0 or all([0 == temp_ans_infl_treshold[x] for x in temp_ans_infl_treshold]):
                        temp_ans_infl_treshold = {x:temp_ans_infl[x] for x in temp_ans_infl if temp_ans_infl[x] >= temp_ans_infl_mean}

                    if temp_ans[i] in temp_ans_infl and temp_ans[i] in temp_ans_infl_treshold:
                        if temp_ans[i] in predicted:
                            if cl in predicted[temp_ans[i]]["class"]:
                                predicted[temp_ans[i]]["infl"][cl] = temp_ans_infl_treshold[temp_ans[i]]*outputFormats_infl[cl]/len(outputFormats)
                                
                            else:    
                                length = len(predicted[temp_ans[i]]["class"])

                                predicted[temp_ans[i]]["class"].append(cl)
                                predicted[temp_ans[i]]["infl"].setdefault(cl, temp_ans_infl_treshold[temp_ans[i]]*outputFormats_infl[cl]/len(outputFormats))
                        
                        else:
                            predicted.setdefault(temp_ans[i], {"class":[cl], "infl": {cl: temp_ans_infl_treshold[temp_ans[i]]*outputFormats_infl[cl]/len(outputFormats)}})

                    que_val = temp_que[i]
                    ans_val = temp_que[i]
                    rel_index = self.getRelationIntersect(que_val, ans_val)
            
        # using accumulated gradients
        aGradients = self.ARM["aGradients"]
        if len(aGradients) > 1:
            aGradients_predictions = {}
            for sx in que_sessions:
                session_spot = " ".join(sx.split(".")[:-1]).strip()
                
                if session_spot in aGradients:
                    
                    idx = ".".join(sx.split(".")[:-1])+"."+str(int(sx.split(".")[-1]) + aGradients[session_spot])
                    
                    try:
                        memory_id = self.getDataIndexFromSession(idx)

                    except Exception as e:
                        # idx = sx.split(".")[0]+"."+str(int(sx.split(".")[1])+1)+".0"
                        # memory_id = self.getDataIndexFromSession(idx)

                        # try:
                        #     idx = str(int(sx.split(".")[1])+1)+".0.0"
                        #     memory_id = self.getDataIndexFromSession(idx)

                        # except Exception as e:
                        #     pass
                        pass

                    val = self.locateMemoryData(memory_id, "text")
                    aGradients_predictions.setdefault(session_spot, val)
            print("here ========>", aGradients_predictions, self.ARM["confidence"])
            predictions = [xx for xx in aGradients_predictions.values()]
            aGradients_predictions_reverse = {prediction:[] for prediction in set(predictions)}

            # reverse gradients dictionary
            for aGradients_prediction in aGradients_predictions:
                aGradients_predictions_reverse[aGradients_predictions[aGradients_prediction]].append(aGradients_prediction)

            aGradients_predictions_reverse_infl = {xx:len(aGradients_predictions_reverse[xx])/len(aGradients_predictions) for xx in aGradients_predictions_reverse}
            aGradients_predictions_pd = self.sortDict(aGradients_predictions_reverse_infl)
            
            print(aGradients_predictions_pd)
            if len(aGradients_predictions_pd) > 0:
                model = self.ARM["model"]
                if self.ARM["seperator"] == "":
                    while "[var]" in model:
                        model = model.replace("[var]", aGradients_predictions_pd[0][0], 1)

                    model = model.replace(" ", "").strip()

                if self.ARM["confidence"] == 1:
                    if model in predicted:
                        predicted[model]["class"].append("[[ARMclass]]")
                        if "[[ARMclass]]" in predicted[model]["infl"]:
                            predicted[model]["infl"]["[[ARMclass]]"] = (predicted[model]["infl"]["[[ARMclass]]"]+aGradients_predictions_pd[0][-1])/2

                        else:
                            predicted[model]["infl"] = {"[[ARMclass]]":aGradients_predictions_pd[0][-1]}
                        
                    else:
                        predicted.setdefault(model, {"infl":{"[[ARMclass]]":aGradients_predictions_pd[0][-1]}})
                        predicted[model].setdefault("class", ["[[ARMclass]]"])

        self.showProcess("predicted = {}".format(predicted))
        reply = self.predict(predicted)
        ret = reply

        # increase confidence in ARM
        print("model before confidence", model)
        if model == reply:
            if self.ARM["confidence"] < 1:
                self.ARM["confidence"] += 0.5

        else:
            self.ARM["confidence"] = 0

        self.ARM = self.ARM_sessionTracker(data, reply, self.ARM.copy())        
        callback = []

        if ret != None:
            # remove non-[var] classes
            if "[[ARMclass]]" in predicted[reply]["class"]:
                predicted[reply]["class"].remove("[[ARMclass]]")

            #list of indexes that hold info on the class that predicted
            for reply_class in predicted[reply]["class"]:
                li = output_classes[reply_class]
                
                #redo of que and ans
                xans = list(set([ans[x] for x in li if ans[x] == reply]))
                xans_sessions = [ans_sessions[x] for x in li]
                
                reque, reans, reinfl, reque_sessions, reans_sessions, rein_memory = self.getQueAns(xans)
                self.xxque, self.xxans, self.xxqna_infl, self.xxque_sessions, self.xxans_sessions, self.xxin_memory = [], [], [], [], [], []
                for i in range(len(reque)):
                    if reque_sessions[i] in xans_sessions:
                        self.xxque.append(reque[i])
                        self.xxans.append(reans[i])
                        self.xxqna_infl.append(reinfl[i])
                        self.xxque_sessions.append(reque_sessions[i])
                        self.xxans_sessions.append(reans_sessions[i])
                        self.xxin_memory.append(rein_memory[i])
                
                # for i in range(len(self.xxque)):
                #     print("xxque = {}, xxans = {}, xxqna_infl = {}, xxque_sessions = {}, xxans_sessions = {}, xxin_memory = {}".format(
                #         self.xxque[i], self.xxans[i], self.xxqna_infl[i], self.xxque_sessions[i], self.xxans_sessions[i], self.xxin_memory[i]))
                
                cb = False
                if len(self.xxans) > 1:
                    cb = True

                callback.append(cb)

        if callback.count(True) > callback.count(False):
            callback = True

        else:
            callback = False

        if callback:
            ret = (ret, callback)
        
        return ret

    def predict(self, predicted):
        pd = self.sortDict({x:sum(predicted[x]["infl"].values()) for x in predicted})
        self.showProcess("predicted replies = {}".format(pd))

        if len(pd) > 0:
            reply = pd[0][0]

        else:
            reply = None
        return reply