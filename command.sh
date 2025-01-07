# cmd for speculative sampling
python main.py --approx_model_name llama1b --target_model_name llama2-7b -b -g 5 -M 60
python main.py --approx_model_name bloom-560m --target_model_name bloom7b -b -g 5 -M 60
python main.py --approx_model_name bloom-560m --target_model_name bloom7b --approx_tmp 0.01 -s 77 -b -g 5 -M 60
python main.py --approx_model_name bloom-560m --target_model_name bloom7b --approx_tmp 10 -s 77 -b -g 5 -M 60
python main.py --approx_model_name bloom-560m --target_model_name bloom7b --approx_tmp 0.01 -s 77 -b -g 7 -M 60
python main.py --approx_model_name llama1b --target_model_name llama2-7b --approx_tmp 0.01 -s 77 -b -g 15 -M 60

# cmd for batch test
python batching.py --model_name bloom7b -g 10 -t 20 -b 1
python batching.py --model_name bloom7b -g 20 --step 2 -t 20 -b 1
python batching.py --model_name llama2-7b -g 30 --step 2 -t 50 -b 1