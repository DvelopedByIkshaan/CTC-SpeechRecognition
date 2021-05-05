#!/usr/bin/env bash
echo "Making release for CTC_SpeechRecognition-$1"

python setup.py bdist_wheel
gpg --detach-sign -a dist/CTC_SpeechRecognition-$1-*.whl
twine upload dist/CTC_SpeechRecognition-$1-*.whl dist/CTC_SpeechRecognition-$1-*.whl.asc
