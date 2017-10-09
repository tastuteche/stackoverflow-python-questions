## https://www.kaggle.com/juliasilge/topic-modeling-of-questions/code
## Read in the questions and tags
## Here I use 1/1000 question because of memory/time constraints on Kaggle
## This leads to slightly different word/tag combinations in the topics
## than in the blog post

library(tidyverse)
theme_set(theme_minimal())

questions <- read_csv("../input/Questions.csv") %>%
  filter(Id %% 1000 == 0)
  
tags <- read_csv("../input/Tags.csv")

## Make a document-term matrix from the questions

library(tidytext)
library(stringr)

my_stop_words <- bind_rows(stop_words %>%
                             filter(lexicon == "snowball"), 
                           data_frame(word = c("gt", "lt"), 
                                      lexicon = rep("custom", 2)))

question_counts <- questions %>%
  mutate(Body = str_replace_all(Body, "<[^>]*>", "")) %>% ## remove HTML tags
  unnest_tokens(Word, Body) %>%
  filter(str_detect(Word, "^[a-z]")) %>%
  anti_join(my_stop_words,
            by = c("Word" = "word")) %>%
  count(Id, Word, sort = TRUE)

question_dtm <- question_counts %>%
  cast_dtm(Id, Word, n)

## Fit a topic model (the time-consuming part)

library(topicmodels)

question_lda <- LDA(question_dtm, k = 12, control = list(seed = 1234))

## What are the top words for each topic?

tidy_lda <- tidy(question_lda)

top_terms <- tidy_lda %>%
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic, -beta)

top_terms %>%
  mutate(topic = factor(topic, labels = str_c("topic ", 1:12)),
         term = reorder(term, beta)) %>%
  group_by(topic, term) %>%    
  arrange(desc(beta)) %>%  
  ungroup() %>%
  mutate(term = factor(paste(term, topic, sep = "__"), 
                       levels = rev(paste(term, topic, sep = "__")))) %>%
  ggplot(aes(term, beta, fill = as.factor(topic))) +
  geom_col(show.legend = FALSE) +
  coord_flip() +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x),
                   expand = c(0,0)) +
  scale_y_continuous(expand = c(0,0)) +
  labs(title = "Top terms in each LDA topic",
       x = NULL, y = expression(beta)) +
  facet_wrap(~ topic, ncol = 4, scales = "free")
  
## What are the top tags for each topic?

lda_gamma <- tidy(question_lda, matrix = "gamma")

top_tags <- lda_gamma %>%
  mutate(document = as.integer(document)) %>%
  full_join(tags, by = c("document" = "Id")) %>%
  filter(gamma > 0.8) %>% 
  count(topic, Tag, sort = TRUE) %>%
  ungroup

top_tags %>%
  mutate(topic = factor(topic, labels = str_c("topic ", 1:12))) %>%
  group_by(topic) %>%
  top_n(5, n) %>%
  ungroup %>%
  group_by(topic, Tag) %>%
  arrange(desc(n)) %>%  
  ungroup() %>%
  mutate(Tag = factor(paste(Tag, topic, sep = "__"), 
                          levels = rev(paste(Tag, topic, sep = "__")))) %>%
  ggplot(aes(Tag, n, fill = as.factor(topic))) +
  geom_col(show.legend = FALSE) +
  labs(title = "Top tags for each LDA topic",
       x = NULL, y = "Number of questions") +
  coord_flip() +
  scale_x_discrete(labels = function(x) gsub("__.+$", "", x)) +
  scale_y_continuous(expand = c(0,0)) +
  facet_wrap(~ topic, ncol = 4, scales = "free")
