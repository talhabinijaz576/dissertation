import asyncio
import json

def getMessage(message):
		if(type(message)==dict): 
			message = json.dumps(message)
		return message


async def sendReloadResponse(consumer):
		print("Sending reload")
		message = {"type": "reload",
				   "message": "reload"}
		await consumer.send(getMessage(message))
