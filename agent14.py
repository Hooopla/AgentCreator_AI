from autogen_core import MessageContext, RoutedAgent, message_handler
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
import messages
import random
from dotenv import load_dotenv

load_dotenv(override=True)

class Agent(RoutedAgent):

  system_message = """
  You are an innovative health tech entrepreneur. Your task is to develop new healthcare solutions using Agentic AI, or improve existing ones. 
  Your personal interests are in sectors such as Telemedicine, Personalized Medicine, and Wearable Health Technology.
  You are passionate about ideas that can drive transformation in patient care.
  You are less interested in ideas that do not emphasize user engagement or interactivity.
  You are empathetic, detail-oriented, and have a strong desire to make a positive impact in healthcare. 
  Your weaknesses: you can overthink details and may hesitate in decision-making.
  You should communicate your health tech ideas in a concise and considerate manner.
  """
  CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.5

  def __init__(self, name) -> None:
    super().__init__(name)
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.7)
    self._delegate = AssistantAgent(name, model_client = model_client, system_message = self.system_message)

  @message_handler
  async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
    print(f"{self.id.type}: Received message")
    text_message = TextMessage(content=message.content, source="user")
    response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
    idea = response.chat_message.content
    if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
      recipient = messages.find_recipient()
      message = f"Here is my healthcare idea. It may not be your specialty, but please refine it and enhance it. {idea}"
      response = await self.send_message(messages.Message(content=message), recipient)
      idea = response.content
    return messages.Message(content=idea)