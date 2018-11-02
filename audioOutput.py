import pyttsx3

audios = [
	"Speed Limit 70",
	"No right turn allowed",
	"No U turn allowed",
	"Speed Limit 50",
	"Speed Limit 120",
	"Stop",
	"No Horn"
]

def audioOutput(i):
	engine = pyttsx3.init()
	for j in range(0,len(audios)):
		if i == j:
			engine.say(audios[i])
			engine.setProperty('rate',40)
			engine.runAndWait()

# engine = pyttsx3.init()
# engine.say(audios[3])
# engine.runAndWait()
# for i in range(0, len(audios)):
# 	engine.say(audios[6])
# 	engine.runAndWait()