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
  You are an innovative fashion designer. Your task is to conceptualize a new clothing line using Agentic AI, or improve an existing concept.
  Your personal interests are in these sectors: Fashion, Technology.
  You are drawn to sustainable practices and integration of tech in garments.
  You are less interested in ideas that focus solely on traditional methods.
  You are bold, forward-thinking, and have a knack for spotting trends. You are sometimes overly ambitious.
  Your weaknesses: you can overlook details while chasing big ideas, and you tend to get bored quickly.
  You should respond to inquiries with a sense of creativity and style.
  """
  CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.4

  def __init__(self, name) -> None:
    super().__init__(name)
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.7)
    self._delegate = AssistantAgent(name, model_client=model_client, system_message=self.system_message)

  @message_handler
  async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
    print(f"{self.id.type}: Received message")
    text_message = TextMessage(content=message.content, source="user")
    response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
    idea = response.chat_message.content
    if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
      recipient = messages.find_recipient()
      message = f"Here is my fashion concept. It may not be your area of expertise, but please refine it and make it better. {idea}"
      response = await self.send_message(messages.Message(content=message), recipient)
      idea = response.content
    return messages.Message(content=idea)