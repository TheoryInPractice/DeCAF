
import pickle, argparse


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-f', '--file_name', type=str,
        help="pickle filename + path", required=True)
    
    args = vars(parser.parse_args())
    
    input_file = args.get('file_name')
    
    infile = open(input_file,'rb')
    new_dict = pickle.load(infile)
    infile.close()
    
    print(new_dict)


if __name__=="__main__":
    main()
