import json

# 第一段数据（示例）
data1 = '''{"movieMentions": {"154513": "X-Men: First Class (2011)", "99896": "Spider-Man  (2002)", "205163": "Avengers: Infinity War (2018)", "99583": "Iron Man  (2008)", "100821": "X-Men: The Last Stand (2006)"}, "respondentQuestions": {"154513": {"suggested": 1, "seen": 1, "liked": 1}, "99896": {"suggested": 1, "seen": 1, "liked": 1}, "205163": {"suggested": 1, "seen": 0, "liked": 1}, "99583": {"suggested": 0, "seen": 1, "liked": 1}, "100821": {"suggested": 1, "seen": 1, "liked": 1}}, "messages": [{"timeOffset": 0, "text": "What kind of movies do you like", "senderWorkerId": 961, "messageId": 204241, "entity": [], "entity_name": [], "movie": [], "movie_name": [], "word_name": ["movies"]}, {"timeOffset": 22, "text": "have you seen @205163 ?", "senderWorkerId": 961, "messageId": 204242, "entity": [], "entity_name": [], "movie": ["<http://dbpedia.org/resource/Avengers:_Infinity_War>"], "movie_name": ["Avengers: Infinity War (2018)"], "word_name": ["infinity", "war", "avengers"]}, {"timeOffset": 34, "text": "or @99896", "senderWorkerId": 961, "messageId": 204243, "entity": [], "entity_name": [], "movie": ["<http://dbpedia.org/resource/Spider-Man_(2002_film)>"], "movie_name": ["Spider-Man (2002)"], "word_name": []}, {"timeOffset": 42, "text": "Yes and i liked them", "senderWorkerId": 960, "messageId": 204244, "entity": [], "entity_name": [], "movie": [], "movie_name": [], "word_name": [], "emotion": {"like": 0.7446123957633972, "curious": 0.21789564192295074, "happy": 0.030877424404025078, "grateful": 0.006412883754819632, "negative": 0.00016324553871527314, "agreement": 2.2051457563065924e-05, "nostalgia": 1.630979750188999e-05, "surprise": 1.4117695901205707e-09, "neutral": 3.984783891741728e-10}}, {"timeOffset": 54, "text": "Those are awesome", "senderWorkerId": 960, "messageId": 204245, "entity": [], "entity_name": [], "movie": [], "movie_name": [], "word_name": ["awesome"], "emotion": []}, {"timeOffset": 61, "text": "I like that kind of movies", "senderWorkerId": 960, "messageId": 204246, "entity": [], "entity_name": [], "movie": [], "movie_name": [], "word_name": ["movies"], "emotion": []}, {"timeOffset": 74, "text": "@154513", "senderWorkerId": 961, "messageId": 204247, "entity": [], "entity_name": [], "movie": ["<http://dbpedia.org/resource/X-Men:_First_Class>"], "movie_name": ["X-Men: First Class (2011)"], "word_name": ["class"]}, {"timeOffset": 77, "text": "i also watched @99583", "senderWorkerId": 960, "messageId": 204248, "entity": [], "entity_name": [], "movie": ["<http://dbpedia.org/resource/Iron_Man_(2008_film)>"], "movie_name": ["Iron Man (2008)"], "word_name": ["watched", "iron"], "emotion": {"curious": 0.5749605894088745, "like": 0.3586151599884033, "happy": 0.055916156619787216, "negative": 0.0072077917866408825, "grateful": 0.001774033298715949, "nostalgia": 0.0015225510578602552, "agreement": 2.942771516245557e-06, "surprise": 6.34283708222938e-07, "neutral": 1.0608823686197866e-07}}, {"timeOffset": 81, "text": "those are good movies as well", "senderWorkerId": 961, "messageId": 204249, "entity": ["<http://dbpedia.org/resource/Film>"], "entity_name": ["movies"], "movie": [], "movie_name": [], "word_name": ["movies"]}, {"timeOffset": 84, "text": "No, i have not watched that one", "senderWorkerId": 960, "messageId": 204250, "entity": [], "entity_name": [], "movie": [], "movie_name": [], "word_name": ["watched"], "emotion": {"curious": 0.6738286018371582, "like": 0.3112400472164154, "happy": 0.013210082426667213, "negative": 0.0011006654240190983, "grateful": 0.0004814324202015996, "nostalgia": 0.0001391697151120752, "agreement": 1.3112342323040593e-08, "surprise": 1.4698876560359508e-09, "neutral": 1.947688815784332e-10}}, {"timeOffset": 96, "text": "i think i will try that one", "senderWorkerId": 960, "messageId": 204251, "entity": [], "entity_name": [], "movie": [], "movie_name": [], "word_name": [], "emotion": []}, {"timeOffset": 97, "text": "@100821 was another good xmen movie", "senderWorkerId": 961, "messageId": 204252, "entity": [], "entity_name": [], "movie": ["<http://dbpedia.org/resource/X-Men:_The_Last_Stand>"], "movie_name": ["X-Men: The Last Stand (2006)"], "word_name": ["stand", "movie"]}, {"timeOffset": 111, "text": "okay thanks!", "senderWorkerId": 961, "messageId": 204253, "entity": [], "entity_name": [], "movie": [], "movie_name": [], "word_name": []}, {"timeOffset": 113, "text": "Oooh, so there are many", "senderWorkerId": 960, "messageId": 204254, "entity": [], "entity_name": [], "movie": [], "movie_name": [], "word_name": ["oooh"], "emotion": {"grateful": 0.5214054584503174, "happy": 0.41092735528945923, "like": 0.04076748713850975, "curious": 0.02582777850329876, "negative": 0.0006225014221854508, "agreement": 0.0004388616362120956, "nostalgia": 9.594837138138246e-06, "surprise": 5.778077820650651e-07, "neutral": 3.955661043164582e-07}}, {"timeOffset": 116, "text": "thanks", "senderWorkerId": 960, "messageId": 204255, "entity": [], "entity_name": [], "movie": [], "movie_name": [], "word_name": [], "emotion": []}, {"timeOffset": 126, "text": "Good bye", "senderWorkerId": 960, "messageId": 204256, "entity": [], "entity_name": [], "movie": [], "movie_name": [], "word_name": ["bye"], "emotion": []}], "conversationId": "20051", "respondentWorkerId": 961, "initiatorWorkerId": 960, "initiatorQuestions": {"154513": {"suggested": 0, "seen": 0, "liked": 1}, "99896": {"suggested": 1, "seen": 1, "liked": 1}, "205163": {"suggested": 1, "seen": 1, "liked": 1}, "99583": {"suggested": 1, "seen": 1, "liked": 1}, "100821": {"suggested": 1, "seen": 0, "liked": 1}}}'''

# 第二段数据（示例）
data2 = '''{
  "conversationId": "20051",
  "messages": [
    {
      "timeOffset": 0,
      "text": "What kind of movies do you enjoy?",
      "senderWorkerId": 961,
      "messageId": 204241,
      "movieMentions": []
    },
    {
      "timeOffset": 22,
      "text": "Have you seen @205163?",
      "senderWorkerId": 961,
      "messageId": 204242,
      "movieMentions": ["<http://dbpedia.org/resource/Avengers:_Infinity_War>"]
    },
    {
      "timeOffset": 34,
      "text": "Or @99896?",
      "senderWorkerId": 961,
      "messageId": 204243,
      "movieMentions": ["<http://dbpedia.org/resource/Spider-Man_(2002_film)>"]
    },
    {
      "timeOffset": 42,
      "text": "Yes, I’ve watched both and I really liked them!",
      "senderWorkerId": 960,
      "messageId": 204244,
      "movieMentions": []
    },
    {
      "timeOffset": 54,
      "text": "They’re awesome!",
      "senderWorkerId": 960,
      "messageId": 204245,
      "movieMentions": []
    },
    {
      "timeOffset": 61,
      "text": "I enjoy movies like those.",
      "senderWorkerId": 960,
      "messageId": 204246,
      "movieMentions": []
    },
    {
      "timeOffset": 74,
      "text": "@154513, I also watched that one.",
      "senderWorkerId": 961,
      "messageId": 204247,
      "movieMentions": ["<http://dbpedia.org/resource/X-Men:_First_Class>"]
    },
    {
      "timeOffset": 77,
      "text": "I’ve also seen @99583.",
      "senderWorkerId": 960,
      "messageId": 204248,
      "movieMentions": ["<http://dbpedia.org/resource/Iron_Man_(2008_film)>"]
    },
    {
      "timeOffset": 81,
      "text": "Those are good movies as well!",
      "senderWorkerId": 961,
      "messageId": 204249,
      "movieMentions": []
    },
    {
      "timeOffset": 84,
      "text": "No, I haven’t watched that one yet.",
      "senderWorkerId": 960,
      "messageId": 204250,
      "movieMentions": []
    },
    {
      "timeOffset": 96,
      "text": "I think I’ll give it a try soon!",
      "senderWorkerId": 960,
      "messageId": 204251,
      "movieMentions": []
    },
    {
      "timeOffset": 97,
      "text": "@100821 was another great X-Men movie.",
      "senderWorkerId": 961,
      "messageId": 204252,
      "movieMentions": ["<http://dbpedia.org/resource/X-Men:_The_Last_Stand>"]
    },
    {
      "timeOffset": 111,
      "text": "Thanks for the recommendations!",
      "senderWorkerId": 961,
      "messageId": 204253,
      "movieMentions": []
    },
    {
      "timeOffset": 113,
      "text": "Oh, so there are many X-Men movies!",
      "senderWorkerId": 960,
      "messageId": 204254,
      "movieMentions": []
    },
    {
      "timeOffset": 116,
      "text": "Thanks again!",
      "senderWorkerId": 960,
      "messageId": 204255,
      "movieMentions": []
    },
    {
      "timeOffset": 126,
      "text": "Goodbye!",
      "senderWorkerId": 960,
      "messageId": 204256,
      "movieMentions": []
    }
  ],
  "respondentWorkerId": 961,
  "initiatorWorkerId": 960
}'''

# 解析为字典
data1_dict = json.loads(data1)
data2_dict = json.loads(data2)

# 遍历 data2_dict 的 messages，使用 messageId 对比
for message in data1_dict["messages"]:
    for msg2 in data2_dict["messages"]:
        if message["messageId"] == msg2["messageId"]:
            message["text"] = msg2["text"]

# 输出更新后的数据
print(json.dumps(data1_dict, indent=2))
