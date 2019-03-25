import argparse
import datetime
import platform

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_epoch", type=int, default="0", help="input model epoch")
    parser.add_argument("--cuda_number", type=int, default="0", help="specify cuda number")
    parser.add_argument("--train_batch_size", type=int, default="32", help="input train batch size")
    parser.add_argument("--test_batch_size", type=int, default="64", help="input test batch size")

    ##データのオプション
    parser.add_argument("--src_length", type=int, default="40", help="input train batch size")
    parser.add_argument("--tgt_length", type=int, default="20", help="input test batch size")
    parser.add_argument("--model_version", type=int, default="1", help="input test batch size")
    parser.add_argument("--use_interro", type=bool, default=False, help="input test batch size")
    parser.add_argument("--data_rate", type=float, default="1", help="input epoch number")

    #model_hyper_parameter
    parser.add_argument("--embed_size", type=int, default="300", help="input rnn hidden size")
    parser.add_argument("--hidden_size", type=int, default="600", help="input rnn hidden size")
    parser.add_argument("--epoch_num", type=int, default="50", help="input epoch number")
    parser.add_argument("--dropout", type=float, default="0.3", help="input epoch number")
    parser.add_argument("--layer_size", type=int, default="2", help="input epoch number")
    parser.add_argument("--vocab_size", type=int, default="30000", help="input epoch number")
    parser.add_argument("--lr", type=float, default="0.001", help="input epoch number")
    parser.add_argument("--teacher_rate", type=float, default="1.0", help="input epoch number")

    #そのたのパラメーター
    parser.add_argument("--print_iter", type=int, default="100", help="input epoch number")
    parser.add_argument("--not_train", type=bool, default=False, help="input epoch number")
    parser.add_argument("--use_train_data", type=bool, default=False, help="input epoch number")
    parser.add_argument("--model_name", type=str, default="", help="input epoch number")
    parser.add_argument("--include_pad", type=bool, default=False, help="input epoch number")
    parser.add_argument("--beam", type=bool, default=False, help="input epoch number")
    parser.add_argument("--beam_width", type=int, default=3, help="input epoch number")
    parser.add_argument("--print_example", type=int, default=0, help="input epoch number")
    args = parser.parse_args()
    args.start_time=str(datetime.datetime.today()).replace(" ","-")
    args.high_epoch=0
    args.high_score=0
    args.system=platform.system()

    return args
