from .models import UnansweredQuestion
import ast

if __name__ == "__main__":
    print("Rag against the machine initialized!\n")
    tree = ast.parse("models.py")
    print(ast.dump(tree, indent=4))
    question = UnansweredQuestion(question="who is the owner of Gala gadir?")
    print(f"id: {question.question_id}    question: {question.question}")
