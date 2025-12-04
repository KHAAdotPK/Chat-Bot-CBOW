/*
    usage/src/process.hh
    Q@khaa.pk
 */

#include <algorithm> // For std::swap 
#include "main.hh" 

#ifndef PROCESS_READ_TRAINED_CBOW_WEIGHTS_TEST_APP_HH
#define PROCESS_READ_TRAINED_CBOW_WEIGHTS_TEST_APP_HH

template <typename E = double>
void cleanup (struct prompt<double>* head) 
{
    struct prompt<double>* ptr = head;

    while (ptr != NULL)
    {
        struct prompt<double>* foo = ptr;

        // We don't deallocate cptr and lptr here because they are managed elsewhere (in the Corpus object)
        // We donot own the memory for cptr and lptr, so we should not deallocate them here.
        /*if (foo->lptr == NULL && foo->cptr != NULL)
        {
            //cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(foo->cptr), sizeof(COMPOSITE));
        }*/

        if (foo->similarity_head != NULL)
        {
            struct prompt<double>* sim_ptr = foo->similarity_head;

            while (sim_ptr != NULL)
            {
                struct prompt<double>* sim_foo = sim_ptr;
                sim_ptr = sim_ptr->next;

                cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(sim_foo), sizeof(struct prompt<double>));                
            }
        }

        if (foo->similarity_target_head != NULL)
        {
            struct prompt<double>* sim_ptr = foo->similarity_target_head;

            while (sim_ptr != NULL)
            {
                struct prompt<double>* sim_foo = sim_ptr;
                sim_ptr = sim_ptr->next;

                cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(sim_foo), sizeof(struct prompt<double>));
            }
        }

        ptr = ptr->next;

        cc_tokenizer::allocator<char>().deallocate(reinterpret_cast<char*>(foo), sizeof(struct prompt<double>));
    }
}

template <typename E = double>
void traverse_context_similarity(struct prompt<E>* head, CORPUS& vocab) throw (ala_exception)
{
    struct prompt<E>* ptr = head;
    struct prompt<E>* similarity_head_ptr = NULL;

    while (ptr != NULL)
    {
        similarity_head_ptr = ptr->similarity_head;

        std::cout<< "Similarities for word: " << ptr->cptr->str.c_str(); 

        if (ptr->cptr != NULL && ptr->lptr != NULL)
        {
            std::cout<< " (" << ptr->lptr->l << "," << ptr->lptr->t << ")" << std::endl;
        }
        else 
        {
            std::cout<< " (OOV)";
        }

        while (similarity_head_ptr != NULL)
        {

            //std::cout<< ptr->cptr->str.c_str() << "(" << ptr->lptr->l << "," << ptr->lptr->t << "): ";

            std::cout<< similarity_head_ptr->cptr->str.c_str() << "(" << similarity_head_ptr->lptr->l << "," << similarity_head_ptr->lptr->t << ")" << ": " << similarity_head_ptr->result << ", ";

            similarity_head_ptr = similarity_head_ptr->next;
        }

        ptr = ptr->next;

        std::cout<< std::endl << std::endl;
    }
}

template <typename E = double>
void traverse_context_similarity_W2 (struct prompt<E>* head, CORPUS& vocab) throw (ala_exception)
{
    struct prompt<E>* ptr = head;
    struct prompt<E>* similarity_head_ptr = NULL;

    while (ptr != NULL)
    {        
        similarity_head_ptr = ptr->similarity_target_head;

        std::cout<< "Similarities for word: " << ptr->cptr->str.c_str(); 

        if (ptr->cptr != NULL && ptr->lptr != NULL)
        {
            std::cout<< " (" << ptr->lptr->l << "," << ptr->lptr->t << ")" << std::endl;
        }
        else 
        {
            std::cout<< " (OOV)";
        }

        while (similarity_head_ptr != NULL)
        {

            //std::cout<< ptr->cptr->str.c_str() << "(" << ptr->lptr->l << "," << ptr->lptr->t << "): ";

            std::cout<< similarity_head_ptr->cptr->str.c_str() << "(" << similarity_head_ptr->lptr->l << "," << similarity_head_ptr->lptr->t << ")" << ": " << similarity_head_ptr->result_similarity_target << ", ";

            similarity_head_ptr = similarity_head_ptr->next;
        }

        ptr = ptr->next;

        std::cout<< std::endl << std::endl;
    }
}

template <typename E = double>
void similarity (Collective<E>& W, struct prompt<double>* head, CORPUS& vocab, bool verbose = false) throw (ala_exception)
{
    struct prompt<double>* ptr = head;
    struct prompt<double>* similarity_head_ptr = NULL, *current_similarity_head_ptr = NULL;

    Collective<E> u, v;

    if (verbose)
    {
        std::cout<< "-:Similarity W1:-" << std::endl;
    }

    while (ptr != NULL)
    {
        E aggregate_validation_loss = 0;

        if (ptr->lptr != NULL)
        {
            v = W.slice(ptr->j*W.getShape().getNumberOfColumns(), W.getShape().getNumberOfColumns());

            if (verbose)
            {
                std::cout<< ptr->cptr->str.c_str() << "(" << ptr->lptr->l << "," << ptr->lptr->t << "): ";
            }

            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < /*W.getShape().getNumberOfRows() -> */ vocab.numberOfUniqueTokens(); i++)
            {
                u = W.slice(i*W.getShape().getNumberOfColumns(), W.getShape().getNumberOfColumns());
                      
                E result = Numcy::Spatial::Distance::cosine<E>(u, v);
                aggregate_validation_loss = aggregate_validation_loss + (1 - result);

                if (verbose)
                {
                    std::cout<< "(" << vocab[i + INDEX_ORIGINATES_AT_VALUE].c_str() << ") " << result << ", ";
                }

                try 
                {
                    if (similarity_head_ptr == NULL)
                    {
                        similarity_head_ptr = reinterpret_cast<struct prompt<double>*>(cc_tokenizer::allocator<char>().allocate(sizeof(struct prompt<E>)));

                        similarity_head_ptr->j = i;
                        similarity_head_ptr->n = 1;
                        similarity_head_ptr->result = result;
                        similarity_head_ptr->cptr = vocab.get_composite_ptr(i + INDEX_ORIGINATES_AT_VALUE, false);
                        similarity_head_ptr->lptr = similarity_head_ptr->cptr->ptr; 
                        similarity_head_ptr->next = NULL;
                        similarity_head_ptr->prev = NULL;

                        ptr->similarity_head = similarity_head_ptr;
                        ptr->n = 1;

                        current_similarity_head_ptr = similarity_head_ptr;
                    }
                    else
                    {
                        current_similarity_head_ptr->next = reinterpret_cast<struct prompt<double>*>(cc_tokenizer::allocator<char>().allocate(sizeof(struct prompt<E>)));
                        current_similarity_head_ptr->next->next = NULL;
                        current_similarity_head_ptr->next->prev = current_similarity_head_ptr;
                        current_similarity_head_ptr = current_similarity_head_ptr->next;
                        current_similarity_head_ptr->j = i;
                        current_similarity_head_ptr->n = 1;
                        current_similarity_head_ptr->result = result;
                        current_similarity_head_ptr->cptr = vocab.get_composite_ptr(i + INDEX_ORIGINATES_AT_VALUE, false);
                        current_similarity_head_ptr->lptr = current_similarity_head_ptr->cptr->ptr;
                        
                        ptr->n = ptr->n + 1;
                    }
                }
                catch (ala_exception& e)
                {

                }
            }

            if (verbose)
            {
                std::cout<< std::endl;

                std::cout<< "Validation Loss = " << aggregate_validation_loss / W.getShape().getNumberOfColumns() << std::endl;
            }
        }
        else
        {
            if (verbose)
            {
                std::cout<< ptr->cptr->str.c_str() << ": (OOV)" << std::endl;
            }

            ptr->similarity_head = NULL;
            ptr->n = 0;
        }

        ptr = ptr->next;

        similarity_head_ptr = NULL;
        current_similarity_head_ptr = NULL;
    }
}

template <typename E = double>
void similarity_w2 (Collective<E>& W1, Collective<E>& W2_t, struct prompt<double>* head, CORPUS& vocab, bool verbose = false) throw (ala_exception)
{
    struct prompt<double>* ptr = head;
    struct prompt<double>* similarity_head_ptr = NULL, *current_similarity_head_ptr = NULL;

    Collective<E> u, v;

    if (verbose)
    {
        std::cout<< "-:Similarity W2:-" << std::endl;
    }

    while (ptr != NULL)
    {
        E aggregate_validation_loss = 0;

        if (ptr->lptr != NULL)
        {
            v = W1.slice(ptr->j*W1.getShape().getNumberOfColumns(), W1.getShape().getNumberOfColumns());

            if (verbose)
            {
                std::cout<< ptr->cptr->str.c_str() << "(" << ptr->lptr->l << "," << ptr->lptr->t << "): ";
            }

            for (cc_tokenizer::string_character_traits<char>::size_type i = 0; i < /*W.getShape().getNumberOfRows() -> */ vocab.numberOfUniqueTokens(); i++)
            {
                u = W2_t.slice(i*W2_t.getShape().getNumberOfColumns(), W2_t.getShape().getNumberOfColumns());
                      
                E result = Numcy::Spatial::Distance::cosine<E>(u, v);
                aggregate_validation_loss = aggregate_validation_loss + (1 - result);

                if (verbose)
                {
                    std::cout<< "(" << vocab[i + INDEX_ORIGINATES_AT_VALUE].c_str() << ") " << result << ", ";
                }

                try 
                {
                    if (similarity_head_ptr == NULL)
                    {
                        similarity_head_ptr = reinterpret_cast<struct prompt<double>*>(cc_tokenizer::allocator<char>().allocate(sizeof(struct prompt<E>)));

                        similarity_head_ptr->j = i;
                        similarity_head_ptr->n = 1;
                        similarity_head_ptr->result_similarity_target = result;
                        similarity_head_ptr->cptr = vocab.get_composite_ptr(i + INDEX_ORIGINATES_AT_VALUE, false);
                        similarity_head_ptr->lptr = similarity_head_ptr->cptr->ptr; 
                        similarity_head_ptr->next = NULL;
                        similarity_head_ptr->prev = NULL;

                        ptr->similarity_target_head = similarity_head_ptr;
                        ptr->n_target = 1;

                        current_similarity_head_ptr = similarity_head_ptr;
                    }
                    else
                    {
                        current_similarity_head_ptr->next = reinterpret_cast<struct prompt<double>*>(cc_tokenizer::allocator<char>().allocate(sizeof(struct prompt<E>)));
                        current_similarity_head_ptr->next->next = NULL;
                        current_similarity_head_ptr->next->prev = current_similarity_head_ptr;
                        current_similarity_head_ptr = current_similarity_head_ptr->next;
                        current_similarity_head_ptr->j = i;
                        current_similarity_head_ptr->n = 1;
                        current_similarity_head_ptr->result_similarity_target = result;
                        current_similarity_head_ptr->cptr = vocab.get_composite_ptr(i + INDEX_ORIGINATES_AT_VALUE, false);
                        current_similarity_head_ptr->lptr = current_similarity_head_ptr->cptr->ptr;
                        
                        ptr->n_target = ptr->n_target + 1;
                    }
                }
                catch (ala_exception& e)
                {

                }
            }

            if (verbose)
            {
                std::cout<< std::endl;

                std::cout<< "Validation Loss = " << aggregate_validation_loss / W2_t.getShape().getNumberOfColumns() << std::endl;
            }
        }
        else
        {
            ptr->similarity_target_head = NULL;
            ptr->n_target = 0;

            if (verbose)
            {
                std::cout<< ptr->cptr->str.c_str() << ": (OOV)" << std::endl;
            }
        }

        ptr = ptr->next;

        similarity_head_ptr = NULL;
        current_similarity_head_ptr = NULL;
    }
}


template <typename E = double>
void bubble_sort(struct prompt<E>* head)
{
    if (head == nullptr)
    { 
        return;
    }

    struct prompt<E>* ptr = head;

    // Traverse every node in the main list
    while (ptr != nullptr)
    {
        // Get the similarity sublist to sort
        struct prompt<E>* sublist = ptr->similarity_head;
        if (sublist == nullptr) 
        {
            ptr = ptr->next;
            continue;
        }

        // Now perform bubble sort on this sublist by swapping DATA only
        bool swapped;
        
        do
        {
            swapped = false;
            struct prompt<E>* current = sublist;

            while (current->next != nullptr)
            {
                if (current->result > current->next->result)
                {
                    // Swap ALL data fields between current and current->next
                    std::swap(current->cptr,          current->next->cptr);
                    std::swap(current->lptr,          current->next->lptr);
                    std::swap(current->j,             current->next->j);
                    std::swap(current->result,        current->next->result);
                    std::swap(current->n,             current->next->n);
                    std::swap(current->similarity_head, current->next->similarity_head);
                    // Do NOT swap next/prev — we want structure unchanged!

                    swapped = true;
                }
                current = current->next;
            }
            // Optional optimization: last element is now in place
            // But we don't track it unless needed
        } 
        while (swapped);

        ptr = ptr->next;
    }
}

template <typename E = double>
void bubble_sort_w2 (struct prompt<E>* head)
{
    if (head == nullptr)
    { 
        return;
    }

    struct prompt<E>* ptr = head;

    // Traverse every node in the main list
    while (ptr != nullptr)
    {
        // Get the similarity sublist to sort
        struct prompt<E>* sublist = ptr->similarity_target_head;
        if (sublist == nullptr) 
        {
            ptr = ptr->next;
            continue;
        }

        // Now perform bubble sort on this sublist by swapping DATA only
        bool swapped;
        
        do
        {
            swapped = false;
            struct prompt<E>* current = sublist;

            while (current->next != nullptr)
            {
                if (current->result_similarity_target > current->next->result_similarity_target)
                {
                    // Swap ALL data fields between current and current->next
                    std::swap(current->cptr,          current->next->cptr);
                    std::swap(current->lptr,          current->next->lptr);
                    std::swap(current->j,             current->next->j);
                    std::swap(current->result_similarity_target, current->next->result_similarity_target);
                    std::swap(current->n_target,             current->next->n_target);
                    std::swap(current->similarity_target_head, current->next->similarity_target_head);
                    // Do NOT swap next/prev — we want structure unchanged!

                    swapped = true;
                }
                current = current->next;
            }
            // Optional optimization: last element is now in place
            // But we don't track it unless needed
        } 
        while (swapped);

        ptr = ptr->next;
    }
}

template <typename E = double>
void traverse(Collective<E>& W, const struct prompt<double>* head) throw (ala_exception)
{
    struct prompt<double>* ptr = head;
    
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
void traverse_context_to_target_pairs (Collective<E>& W1, Collective<E>& W2_t, struct prompt<double>* head, CORPUS& vocab) throw (ala_exception)
{
    struct prompt<double>* ptr = head;
    
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
