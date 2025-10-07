"""
Simple Completion Example

This example shows how to use a completion provider for basic text generation.
"""

from src.providers.completion import OpenAIProvider
from config.settings import settings


def main():
    """Run simple completion example"""
    
    # Initialize provider
    print("Initializing OpenAI provider...")
    provider = OpenAIProvider(
        api_key=settings.openai_api_key,
        model="gpt-4"
    )
    
    # Generate completion
    print("\nGenerating completion...")
    prompt = "Explain quantum computing in simple terms"
    response = provider.complete(prompt)
    
    print(f"\nPrompt: {prompt}")
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
