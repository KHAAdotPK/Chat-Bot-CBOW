/*
    usage/src/process.hh
    Q@khaa.pk
 */

#include "main.hh" 

#ifndef PROCESS_READ_TRAINED_CBOW_WEIGHTS_TEST_APP_HH
#define PROCESS_READ_TRAINED_CBOW_WEIGHTS_TEST_APP_HH

template <typename E = double>
void cleanup (PROMPT_PTR head) 
{
    PROMPT_PTR ptr = head;

    while (ptr != NULL)
    {
        PROMPT_PTR foo = ptr;

        if (foo->lptr == NULL && foo->cptr != NULL)
        {
            cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(foo->cptr), sizeof(COMPOSITE));
        }

        ptr = ptr->next;

        cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(foo), sizeof(PROMPT));
    }
}

template <typename E = double>
void similarity (Collective<E>& W, const PROMPT_PTR head, CORPUS& vocab) throw (ala_exception)
{
    PROMPT_PTR ptr = head;
    Collective<E> u, v;

    std::cout<< "-:Similarity W1:-" << std::endl;

    while (ptr != NULL)
    {
        E aggregate_validation_loss = 0;

        if (ptr->lptr != NULL)
        {
            v = W.slice(ptr->j*W.getShape().getNumberOfColumns(), W.getShape().getNumberOfColumns());

            std::cout<< ptr->cptr->str.c_str() << "(" << ptr->lptr->l << "," << ptr->lptr->t << "): ";

            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < /*W.getShape().getNumberOfRows() -> */ vocab.numberOfUniqueTokens(); i++)
            {
                u = W.slice(i*W.getShape().getNumberOfColumns(), W.getShape().getNumberOfColumns());
                      
                E result = Numcy::Spatial::Distance::cosine<E>(u, v);
                aggregate_validation_loss = aggregate_validation_loss + (1 - result);

                std::cout<< "(" << vocab[i + INDEX_ORIGINATES_AT_VALUE].c_str() << ") " << result << ", ";
            }
            std::cout<< std::endl;

            std::cout<< "Validation Loss = " << aggregate_validation_loss / W.getShape().getNumberOfColumns() << std::endl;
        }
        else
        {
            std::cout<< ptr->cptr->str.c_str() << ": (OOV)" << std::endl;
        }

        ptr = ptr->next;
    }
}

template <typename E = double>
void traverse(Collective<E>& W, const PROMPT_PTR head) throw (ala_exception)
{
    PROMPT_PTR ptr = head;
    
    while (ptr != NULL)
    {
        if (ptr->lptr == NULL)
        {            
            std::cout<< ptr->cptr->str.c_str() << ": (OOV)" << std::endl;

            ptr = ptr->next;            
        }
        else
        {
            Collective<E> word_embedding;
            COMPOSITE_PTR cptr = ptr->cptr;
            LINETOKENNUMBER_PTR lptr = ptr->lptr;
            cc_tokenizer::string_character_traits<char>::size_type j = ptr->j;

            std::cout<< cptr->str.c_str() << ": " << cptr->index << "(IDXoriginate@INDEX_ORIGINATE_AT_VALUE) " << cptr->n_ptr << "#instances " << j << "(IDXunique&Originate@0)" << std::endl;
            try 
            {        
                word_embedding = W.slice(j*W.getShape().getNumberOfColumns(), W.getShape().getNumberOfColumns());
                std::cout<< "Word Embedding: ";
                for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < word_embedding.getShape().getNumberOfColumns(); i++)
                {
                    std::cout<< word_embedding[i] << " ";
                }
                std::cout<< std::endl;
            }
            catch (ala_exception& e)
            {
                throw ala_exception(cc_tokenizer::String<char>("traverse() -> ") + e.what());
            }

            ptr = ptr->next;

            while (lptr != NULL)
            {
                std::cout<< "--> " << lptr->index << "(IDXredundant&Originate@INDEX_ORIGINATE_AT_VALUE) l#" << lptr->l << ",t#" << lptr->t << std::endl;

                lptr = lptr->next;
            }
        }
    }

    /*LINETOKENNUMBER_PTR current = NULL;

    COMPOSITE_PTR cptr = head->cptr;
    LINETOKENNUMBER_PTR lptr = head->lptr;
    
    std::cout<< cptr->str.c_str() << " " << cptr->index << "(IDXunique&Originate@INDEX_ORIGINATE_AT_VALUE) " << cptr->n_ptr << "#instances" << std::endl;

    current = lptr;

    while (current != NULL)
    {                
        std::cout<< "--> " << current->index << "(IDXredundant&Originate@INDEX_ORIGINATE_AT_VALUE) " << " " << current->j <<std::endl;

        current = current->next;
    }*/
}

template <typename E = double>
void traverse_context_to_target_pairs (Collective<E>& W1, Collective<E>& W2_t, const PROMPT_PTR head, CORPUS& vocab) throw (ala_exception)
{
    PROMPT_PTR ptr = head;
    
    std::cout<< "-:Similarity W2:-" << std::endl;

    while (ptr != NULL)
    {
        Collective<E> u, v; 
        
        if (ptr->lptr != NULL)
        {
            std::cout<< ptr->cptr->str.c_str() << ": ";

            u = W1.slice(ptr->j*W1.getShape().getNumberOfColumns(), W1.getShape().getNumberOfColumns());

            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < /*W2_t.getShape().getNumberOfRows() -> */ vocab.numberOfUniqueTokens(); i++)
            {                
                v = W2_t.slice(i*W2_t.getShape().getNumberOfColumns(), W2_t.getShape().getNumberOfColumns());
                      
                E result = Numcy::Spatial::Distance::cosine<E>(u, v);
                
                std::cout<< "(" << vocab[i + INDEX_ORIGINATES_AT_VALUE].c_str() << ") " << result << ", ";
            }            
        }
        else
        {            
            std::cout<< ptr->cptr->str.c_str() << ": (OOV)";
        }

        std::cout<< std::endl << ":-" << std::endl;

        ptr = ptr->next;
    }
}

#endif
