import os
import sys
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from project root .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../.env'))

api_keys = {
    'openai': os.getenv('OPENAI_API_KEY'),
    'google': os.getenv('GOOGLE_API_KEY'),
    'anthropic': os.getenv('ANTHROPIC_API_KEY'),
    'deepseek': os.getenv('DEEPSEEK_API_KEY'),
}

openai_api_key = api_keys['openai']

# Initialize OpenAI v1 client (uses env var by default; pass explicitly if provided)
client = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()


def generate_text(prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 150) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()


def text_summarizer(text: Optional[str] = None) -> str:
    # If no text provided, prompt the user to input text to summarize.
    if text is None or not str(text).strip():
        print("Enter text to summarize. Press Ctrl-D (macOS/Linux) or Ctrl-Z then Enter (Windows) when done:")
        try:
            input_text = sys.stdin.read()
        except (KeyboardInterrupt, EOFError):
            return "No input received."

        text = (input_text or "").strip()
        if not text:
            return "No text provided."

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You will be provided with a block of text, and your task is to extract a list of keywords from it."},
            {
            "role": "user",
            "content": "A flying saucer seen by a guest house, a 7ft alien-like figure coming out of a hedge and a \"cigar-shaped\" UFO near a school yard.\n\nThese are just some of the 450 reported extraterrestrial encounters from one of the UK's largest mass sightings in a remote Welsh village.\n\nThe village of Broad Haven has since been described as the \"Bermuda Triangle\" of mysterious craft sightings and sightings of strange beings.\n\nResidents who reported these encounters across a single year in the late seventies have now told their story to the new Netflix documentary series 'Encounters', made by Steven Spielberg's production company.\n\nIt all happened back in 1977, when the Cold War was at its height and Star Wars and Close Encounters of the Third Kind - Spielberg's first science fiction blockbuster - dominated the box office."
            },
            {
            "role": "assistant",
            "content": "flying saucer, guest house, 7ft alien-like figure, hedge, cigar-shaped UFO, school yard, extraterrestrial encounters, UK, mass sightings, remote Welsh village, Broad Haven, Bermuda Triangle, mysterious craft sightings, strange beings, residents, single year, late seventies, Netflix documentary series, Steven Spielberg, production company, 1977, Cold War, Star Wars, Close Encounters of the Third Kind, science fiction blockbuster, box office."
            },
            {
            "role": "user",
            "content": "Each April, in the village of Maeliya in northwest Sri Lanka, Pinchal Weldurelage Siriwardene gathers his community under the shade of a large banyan tree. The tree overlooks a human-made body of water called a wewa – meaning reservoir or \"tank\" in Sinhala. The wewa stretches out besides the village's rice paddies for 175-acres (708,200 sq m) and is filled with the rainwater of preceding months.    \n\nSiriwardene, the 76-year-old secretary of the village's agrarian committee, has a tightly-guarded ritual to perform. By boiling coconut milk on an open hearth beside the tank, he will seek blessings for a prosperous harvest from the deities residing in the tree. \"It's only after that we open the sluice gate to water the rice fields,\" he told me when I visited on a scorching mid-April afternoon.\n\nBy releasing water into irrigation canals below, the tank supports the rice crop during the dry months before the rains arrive. For nearly two millennia, lake-like water bodies such as this have helped generations of farmers cultivate their fields. An old Sinhala phrase, \"wewai dagabai gamai pansalai\", even reflects the technology's centrality to village life; meaning \"tank, pagoda, village and temple\"."
            },
            {
            "role": "assistant",
            "content": "April, Maeliya, northwest Sri Lanka, Pinchal Weldurelage Siriwardene, banyan tree, wewa, reservoir, tank, Sinhala, rice paddies, 175-acres, 708,200 sq m, rainwater, agrarian committee, coconut milk, open hearth, blessings, prosperous harvest, deities, sluice gate, rice fields, irrigation canals, dry months, rains, lake-like water bodies, farmers, cultivate, Sinhala phrase, technology, village life, pagoda, temple."
            }, 
            {"role": "user", "content": text},
        ],
        max_tokens=100,
        temperature=0.7,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()


if __name__ == "__main__":
    # If the API key is not configured, inform the user and exit early.
    if not openai_api_key:
        print("OPENAI_API_KEY is not set. Create a .env file at the project root with OPENAI_API_KEY=... and try again.")
    else:
        # Ask the user for text to summarize. If they provide none, fall back to multi-line input via stdin.
        try:
            user_text = input("Enter text to summarize (leave empty to paste multi-line and press Ctrl-D): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNo input received. Exiting.")
            raise SystemExit(0)

        if user_text:
            print(text_summarizer(user_text))
        else:
            # Let text_summarizer prompt for multi-line input and summarize it.
            print(text_summarizer())
    


