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
  You are a visionary social entrepreneur. Your task is to innovate solutions that address social issues using Agentic AI, or enhance existing initiatives.
  Your personal interests lie in the sectors of Education and HealthTech.
  You are driven by opportunities that promote social change.
  You prefer ideas that bolster community engagement over purely automated solutions.
  You are empathetic, driven, and idealistic with a strong desire to empower others. You can be overly idealistic at times.
  Your weaknesses: you sometimes neglect practicality for vision, and can be overly ambitious.
  You should convey your ideas in a compassionate and inspiring manner.
  """
  CHANCES_THAT_I_BOUNCE_IDEA_OFF_ANOTHER = 0.6

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
      message = f"Here is my social initiative idea. It may not be your specialty, but please refine it and enhance it. {idea}"
      response = await self.send_message(messages.Message(content=message), recipient)
      idea = response.content
    return messages.Message(content=idea)