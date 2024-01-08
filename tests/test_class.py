'''
pytest tests/test_class.py
pytest -q tests/test_class.py
'''
class TestClass:
    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        class Person():
            age = 10
            name = 'Tim'
        p = Person()
        assert hasattr(p, 'age')