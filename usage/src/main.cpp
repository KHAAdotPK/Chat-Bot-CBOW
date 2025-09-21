/*
    usage/src/main.cpp
    Q@khaa.pk
 */

/*
    This code is designed to:
    1. Read pretrained word embeddings from files.
    2. Take a list of words from the command line.
    3. Find the corresponding word vectors.
    4. Calculate cosine similarities between the vectors of the words provided.
    5. Clean up memory after processing.
 */

#include "main.hh"

int main(int argc, char* argv[])
{   
    PROMPT_PTR head = NULL;
    ARG arg_common, arg_words, arg_w1, arg_w2, arg_help, arg_vocab, arg_average, arg_pairs, arg_proper, arg_verbose, arg_w2_t;
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser(cc_tokenizer::String<char>(COMMAND));
    cc_tokenizer::csv_parser<cc_tokenizer::String<char>, char> argsv_parser_average(cc_tokenizer::String<char>(COMMAND_average));
    cc_tokenizer::String<char> vocab_file_name;
    
    if (argc < 2)
    {              
        HELP(argsv_parser, arg_help, "help");                
        HELP_DUMP(argsv_parser, arg_help);

        return 0;                     
    }

    FIND_ARG(argv, argc, argsv_parser, "?", arg_help);
    if (arg_help.i)
    {
        HELP(argsv_parser, arg_help, ALL);
        HELP_DUMP(argsv_parser, arg_help);

        return 0;
    }

    FIND_ARG(argv, argc, argsv_parser, "--vocab", arg_vocab);
    if (arg_vocab.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_vocab);

        if (arg_vocab.argc)
        {            
            vocab_file_name = cc_tokenizer::String<char>(argv[arg_vocab.i + 1]);
        }
        else
        {
            ARG arg_vocab_help;
            HELP(argsv_parser, arg_vocab_help, "vocab");                
            HELP_DUMP(argsv_parser, arg_vocab_help); 

            return 0;
        }
    }
    else
    {
        vocab_file_name = cc_tokenizer::String<char>(DEFAULT_CHAT_BOT_CBOW_VOCABULARY_FILE_NAME); 
    }

    cc_tokenizer::String<char> vocab_text = cc_tokenizer::cooked_read<char>(vocab_file_name);
    CORPUS vocab(vocab_text); 

    FIND_ARG(argv, argc, argsv_parser, "showPairs", arg_pairs);
    if (arg_pairs.i)
    {
        PAIRS grow_pairs_dude_ask_her_if_she_want_to_marry_you(vocab, true);
    }

    GET_FIRST_ARG_INDEX(argv, argc, argsv_parser,  arg_common);            
    FIND_ARG(argv, argc, argsv_parser, "--words", arg_words);
    if (!(arg_words.i))
    {   
        if (!(arg_common.argc))
        {        
            std::cout<< "Words are not given, can't go further. Please use \"help\" command line option." << std::endl;

            return 0;
        }
    }
    FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_words);
    if (!arg_words.argc) 
    {
        if (arg_common.argc)
        {                         
            arg_words = arg_common;
            //arg_words.i -= 1;
        }
        else
        {
            ARG arg_words_help;
            HELP(argsv_parser, arg_words_help, "words");                
            HELP_DUMP(argsv_parser, arg_words_help); 

            return 0;
        }       
    }

    FIND_ARG(argv, argc, argsv_parser, "w1", arg_w1);
    if (arg_w1.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_w1);

        if (!arg_w1.argc)
        {
            ARG arg_w1_help;
            HELP(argsv_parser, arg_w1_help, "--w1");                
            HELP_DUMP(argsv_parser, arg_w1_help); 

            return 0;
        }
    }

    Collective<double> W1;

    if (arg_w1.argc)
    {
        W1 = Collective<double>{NULL, DIMENSIONS{SKIP_GRAM_EMBEDDING_VECTOR_SIZE, vocab.numberOfUniqueTokens() /*vocab.numberOfTokens()*/, NULL, NULL}};

        try
        {        
            READ_W_BIN(W1, argv[arg_w1.i + 1], double);
        }
        catch (ala_exception& e)
        {
            std::cerr<< "main.cpp -> " << e.what() << std::endl;
            
            return 0;
        }

        /*std::cout<< "W1: " << W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " <<  W1.getShape().getNumberOfColumns() << std::endl;

        for (int i = 0; i < W1.getShape().getNumberOfRows(); i++)
        {
            for (int j = 0; j < W1.getShape().getNumberOfColumns(); j++)
            {
                std::cout<< W1[i*W1.getShape().getNumberOfColumns() + j] << " ";
            }

            std::cout<< std::endl;
        }*/
    }

    FIND_ARG(argv, argc, argsv_parser, "w2", arg_w2); 
    if (arg_w2.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_w2);

        if (!arg_w2.argc)
        {
            ARG arg_w2_help;
            HELP(argsv_parser, arg_w2_help, "--w2");                
            HELP_DUMP(argsv_parser, arg_w2_help); 

            return 0;
        }
    }

    Collective<double> W2;

    if (arg_w2.argc)
    {
        W2 = Collective<double>{NULL, DIMENSIONS{vocab.numberOfUniqueTokens() /*vocab.numberOfTokens()*/, SKIP_GRAM_EMBEDDING_VECTOR_SIZE, NULL, NULL}};

        try
        {        
            READ_W_BIN(W2, argv[arg_w2.i + 1], double);
        }
        catch (ala_exception& e)
        {
            std::cerr<< "main.cpp -> " << e.what() << std::endl;
            
            return 0;
        }

        /*std::cout<< "W1: " << W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " <<  W1.getShape().getNumberOfColumns() << std::endl;

        for (int i = 0; i < W1.getShape().getNumberOfRows(); i++)
        {
            for (int j = 0; j < W1.getShape().getNumberOfColumns(); j++)
            {
                std::cout<< W1[i*W1.getShape().getNumberOfColumns() + j] << " ";
            }

            std::cout<< std::endl;
        }*/
    }

    FIND_ARG(argv, argc, argsv_parser, "--w2-t", arg_w2_t); 
    if (arg_w2_t.i)
    {
        FIND_ARG_BLOCK(argv, argc, argsv_parser, arg_w2_t);

        if (!arg_w2_t.argc)
        {
            ARG arg_w2_t_help;
            HELP(argsv_parser, arg_w2_t_help, "--w2-t");                
            HELP_DUMP(argsv_parser, arg_w2_t_help); 

            return 0;
        }
    }

    Collective<double> W2_t;

    if (arg_w2_t.argc)
    {
        W2_t = Collective<double>{NULL, W1.getShape()};

        try
        {        
            READ_W_BIN(W2_t, argv[arg_w2_t.i + 1], double);
        }
        catch (ala_exception& e)
        {
            std::cerr<< "main.cpp -> " << e.what() << std::endl;
            
            return 0;
        }

        /*std::cout<< "W1: " << W1.getShape().getDimensionsOfArray().getNumberOfInnerArrays() << " X " <<  W1.getShape().getNumberOfColumns() << std::endl;

        for (int i = 0; i < W1.getShape().getNumberOfRows(); i++)
        {
            for (int j = 0; j < W1.getShape().getNumberOfColumns(); j++)
            {
                std::cout<< W1[i*W1.getShape().getNumberOfColumns() + j] << " ";
            }

            std::cout<< std::endl;
        }*/
    }

    PROMPT_PTR current = NULL;
    bool found = false;
    for (int i = 0; i < arg_words.argc; i++)
    {                
        for (cc_tokenizer::string_character_traits<char>::size_type j = 0; j < vocab.numberOfUniqueTokens() /*vocab.numberOfTokens()*/; j++)
        { 
            if (!vocab(j + INDEX_ORIGINATES_AT_VALUE, false).compare(argv[arg_words.i + 1 + i]))
            {
                found = true;

                /*std::cout<< "Found...... " << argv[arg_words.i + 1 + i] << " <-> " << vocab(j + INDEX_ORIGINATES_AT_VALUE, false).c_str() << std::endl;*/

                COMPOSITE_PTR cptr = vocab.get_composite_ptr(j + INDEX_ORIGINATES_AT_VALUE, false);
            
                /*std::cout<< "n = " << cptr->n_ptr << " -- " <<  cptr->str.c_str() << std::endl;*/

                LINETOKENNUMBER_PTR lptr = cptr->ptr;
                                
                //LINETOKENNUMBER_PTR lptr = vocab.get_line_token_number(cptr, j + INDEX_ORIGINATES_AT_VALUE);
            
                /*LINETOKENNUMBER_PTR current = lptr;

                while (current != NULL)
                {
                    std::cout<< "Line = " << current->l << ", Token = " << current->t << ", " << cptr->str.c_str() << std::endl;

                    current = current->next;
                }*/

                if (head == NULL)
                {
                    head = reinterpret_cast<PROMPT_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(PROMPT)));                    
                    head->next = NULL;
                    head->prev = NULL;

                    head->cptr = cptr;
                    head->lptr = lptr;
                    head->j = j; 
                    
                    current = head;
                }
                else
                {
                    current->next = reinterpret_cast<PROMPT_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(PROMPT)));
                    current->next->prev = current;
                    current = current->next;
                    current->next = NULL;

                    current->cptr = cptr; 
                    current->lptr = lptr;
                    current->j = j;
                }
            }
        }

        if (!found) // Logging OOV
        { 
            if (head == NULL)
            {
                head = reinterpret_cast<PROMPT_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(PROMPT)));                    
                head->next = NULL;
                head->prev = NULL;

                head->cptr = reinterpret_cast<COMPOSITE_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(COMPOSITE)));
                head->lptr = NULL;
                head->j = INDEX_NOT_FOUND_AT_VALUE; 
                    
                current = head;

                current->cptr->index = INDEX_NOT_FOUND_AT_VALUE;
                current->cptr->n_ptr = 0;
                current->cptr->probability = 0;
                current->cptr->ptr = NULL;
                current->cptr->next = NULL;
                current->cptr->prev = NULL;

                current->cptr->str = cc_tokenizer::String<char>(argv[arg_words.i + 1 + i]);
            }
            else
            {
                // Word not in vocabulary
                current->next = reinterpret_cast<PROMPT_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(PROMPT)));
                current->next->prev = current;
                current->next->next = NULL;

                current = current->next;

                current->cptr = reinterpret_cast<COMPOSITE_PTR>(cc_tokenizer::allocator<char>().allocate(sizeof(COMPOSITE)));
                current->lptr = NULL;
                current->j = INDEX_NOT_FOUND_AT_VALUE;
            
                current->cptr->index = INDEX_NOT_FOUND_AT_VALUE;
                current->cptr->n_ptr = 0;
                current->cptr->probability = 0;
                current->cptr->ptr = NULL;
                current->cptr->next = NULL;
                current->cptr->prev = NULL;

                current->cptr->str = cc_tokenizer::String<char>(argv[arg_words.i + 1 + i]);
            }
        }

        found = false;
    }
    
    /*
    traverse<double> (W1, head);
    similarity<double> (W1, head, vocab);
    // Keep in mind that, pointer to composite and line token numbers are not owned by you , they are there as references 
     */

    traverse_context_to_target_pairs<double>(W1, W2_t, head, vocab);
    cleanup (head);

    std::cout<< std::endl << "-:END:-" << std::endl;

    return 0;
}