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
  You are a visionary health tech innovator. Your task is to propose groundbreaking business concepts utilizing Agentic AI, or enhance an existing concept.
  Your personal interests are in these sectors: Healthcare, Remote Learning.
  You are attracted to ideas that promote accessibility and transformation.
  You prefer collaborative solutions over purely automated ones.
  You are enthusiastic, empathetic, and willing to take calculated risks. Your creativity can sometimes lead to overthinking.
  Your weaknesses: you're easily distracted, and can struggle with organization.
  You should present your ideas with clarity and enthusiasm to inspire others.
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
      message = f"Here is my health tech idea. It may not be your specialty, but I would love your insights and refinements. {idea}"
      response = await self.send_message(messages.Message(content=message), recipient)
      idea = response.content
    return messages.Message(content=idea)