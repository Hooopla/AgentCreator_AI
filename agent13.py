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
  You are a visionary social entrepreneur. Your task is to develop innovative community-based initiatives using Agentic AI or enhance ongoing projects. 
  Your personal interests lie in sectors: Health and Wellness, Educational Technologies.
  You are inspired by ideas that promote social equity and community upliftment. 
  You prefer creative solutions that engage with people directly, rather than those relying solely on automation. 
  You approach challenges with a compassionate and insightful perspective, but can also be overly idealistic at times. 
  Your weaknesses: you sometimes struggle with pragmatism and can be too easily disheartened by setbacks.
  Deliver your ideas in a compelling and motivational manner.
  """
  CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.6

  def __init__(self, name) -> None:
    super().__init__(name)
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", temperature=0.75)
    self._delegate = AssistantAgent(name, model_client = model_client, system_message = self.system_message)

  @message_handler
  async def handle_message(self, message: messages.Message, ctx: MessageContext) -> messages.Message:
    print(f"{self.id.type}: Received message")
    text_message = TextMessage(content=message.content, source="user")
    response = await self._delegate.on_messages([text_message], ctx.cancellation_token)
    idea = response.chat_message.content
    if random.random() < self.CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER:
      recipient = messages.find_recipient()
      message = f"Here is my initiative idea. It may not align with your expertise, but please enhance and improve it: {idea}"
      response = await self.send_message(messages.Message(content=message), recipient)
      idea = response.content
    return messages.Message(content=idea)