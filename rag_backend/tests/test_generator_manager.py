import json
import asyncio
from datetime import datetime
from rag_backend.components.managers import GeneratorManager

async def test_qa(generator_manager, rag_config, question, context, log_func):
    """Test a single question-answer pair"""
    log_func(f"\nQuestion: {question}")
    log_func(f"Context: {context}")
    
    try:
        async for response in generator_manager.generate_stream(rag_config, question, context, []):
            if response.get("message"):
                log_func(f"Answer: {response['message']}")
            if response.get("finish_reason") == "stop":
                break
    except Exception as e:
        log_func(f"Error: {str(e)}")
    
    log_func("-" * 80)

def run_tests():
    # Create a log file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"generator_manager_test_results_{timestamp}.txt"
    
    with open(log_file, "w") as f:
        def log(message):
            print(message)
            f.write(message + "\n")
            f.flush()

        log(f"\n=== Testing GeneratorManager QA ===\n")
        log(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        generator_manager = GeneratorManager()

        # Create RAG config
        rag_config = {
            "Generator": type('obj', (object,), {
                'selected': 'Ollama',
                'components': {
                    'Ollama': type('obj', (object,), {
                        'config': {
                            "Model": type('obj', (object,), {'value': 'qwen3:8b'}),
                            "System Message": type('obj', (object,), {'value': 'You are a helpful assistant.'})
                        }
                    })
                }
            })
        }

        # Test cases with questions and context
        test_cases = [
            {
                "question": "What is the capital of France?",
                "context": "Paris is the capital city of France. It is known for the Eiffel Tower and the Louvre Museum."
            },
            {
                "question": "What is the main function of a CPU?",
                "context": "A CPU (Central Processing Unit) is the primary component of a computer that processes instructions and performs calculations. It's often called the brain of the computer."
            },
            {
                "question": "How does photosynthesis work?",
                "context": "Photosynthesis is the process by which plants convert light energy into chemical energy. They use sunlight, water, and carbon dioxide to produce glucose and oxygen."
            }
        ]

        # Run the QA tests
        for i, test_case in enumerate(test_cases, 1):
            log(f"\nTest Case {i}:")
            asyncio.run(test_qa(
                generator_manager,
                rag_config,
                test_case["question"],
                test_case["context"],
                log
            ))

        log(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"Results saved to: {log_file}")

if __name__ == '__main__':
    run_tests() 