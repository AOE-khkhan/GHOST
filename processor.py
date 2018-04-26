class Processor:
    def runInputProcessing(self, data, callback=False):
        que, ans, qna_infl, dataMap, commonFormatQue, commonFormatAns, searchFormats, outputFormats, output_classes, que_sessions, ans_sessions, in_memory = self.getFeatures(data, callback)
        # for i in range(len(que)):                
        #     self.showProcess("\nque = {}\nans = {}\ndatamap = {}\ncommonFormatQue = {}\ncommonFormatAns = {}\noutputFormat = {}\nsearchFormat = {}\n".format(
        #         que[i], ans[i], dataMap[i], commonFormatQue[i], commonFormatAns[i], outputFormats[i], searchFormats[i]))

        #instantiating variables
        predicted = {}

        #first log influence of repeating [var] values
        outputFormats_infl = {x:outputFormats.count(x) for x in set(outputFormats)}

        for clx in output_classes.copy():
            print(outputFormats, outputFormats_infl, clx)
            #check for already defined ans values
            if outputFormats_infl[clx] > 1:
                predicted.setdefault(clx,{"class":clx, "infl": outputFormats_infl[clx]/len(outputFormats)})

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


            self.showProcess("search_memory_li = {}".format(temp_search_memory_li))
            temp_search_memory = max([temp_search_memory_li.count(x) for x in set(temp_search_memory_li)]) == temp_search_memory_li.count(True)
            
            self.showProcess('\nclass: {}'.format(cl))
            for i in range(len(temp_que)):                
                # self.showProcess("\ntemp_que = {}\ntemp_ans = {}\ntemp_datamap = {}\ntemp_commonFormatQue = {}\ntemp_commonFormatAns = {}\ntemp_outputFormat = {}\ntemp_searchFormat = {}\ntemp_que_session_index = {}\ntemp_ans_session_index = {}\ntemp_search_memory = {}\n".format(
                #     temp_que[i], temp_ans[i], temp_dataMap[i], temp_commonFormatQue[i], temp_commonFormatAns[i], temp_outputFormats[i], temp_searchFormats[i], temp_que_session_index[i], temp_ans_session_index[i], temp_search_memory_li[i]))

                if "[var]" == cl:
                    val = temp_que[i]
                    que_rel, que_rel_index = self.getRelated(val, " ", (len(val.split())**-2)/2, strict=False)

                    #check ans in memory
                    val = temp_ans[i]
                    ans_rel, ans_rel_index = self.getRelated(val, " ", (len(val.split())**-2)/2, strict=False)

                    #check intersection of que and ans in memory
                    if len(ans_rel_index) <= len(que_rel_index):
                        val_li = ans_rel
                        val_li2 = que_rel

                    else:
                        val_li = que_rel
                        val_li2 = ans_rel

                    rel_index = {x:val_li[x]*val_li2[x] for x in val_li if x in val_li2}
                    print(rel_index) 
            
            
        self.showProcess("predicted = {}".format(predicted))
        reply = self.predict(predicted)
        ret = reply

        #list of indexes that hold info on the class that predicted
        li = output_classes[predicted[reply]["class"]]
        
        #redo of que and ans
        xans = list(set([ans[x] for x in li]))
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
        
        for i in range(len(self.xxque)):
            print("xxque = {}, xxans = {}, xxqna_infl = {}, xxque_sessions = {}, xxans_sessions = {}, xxin_memory = {}".format(
                self.xxque[i], self.xxans[i], self.xxqna_infl[i], self.xxque_sessions[i], self.xxans_sessions[i], self.xxin_memory[i]))
        
        callback = False
        if len(self.xxans) > 1:
            callback = True

        if callback:
            ret = (ret, callback)
        
        return ret

    def predict(self, predicted):
        pd = self.sortDict({x:predicted[x]["infl"] for x in predicted})
        self.showProcess("predicted replies = {}".format(pd))

        reply = pd[0][0]
        return reply