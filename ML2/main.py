# ML HW 3
from utils import *
import pyfpgrowth
import nltk
from rake_nltk import Rake

# Load Dataset
data = load_data()

####################### Question 1.1 ##############################
supporters = dict()
sorted_confs = sorted(get_values(data, key='conference'))
sorted_years = sorted(get_values(data, key='year'))

# Find supporters for each conference
for conf in sorted_confs:
    filtered_data = get_filtered_data(data, key='conference', values=[conf])

    baskets = get_baskets(filtered_data, 'author')
    # Get frequent  itenset
    patterns = pyfpgrowth.find_frequent_patterns(baskets, 2)
    # Get only 1-itemset
    patterns = dict((k, v) for k, v in patterns.items() if len(k) == 1)
    # Sort by frequency
    patterns = get_sorted_patterns(patterns)
    # Get top 5 supporters
    patterns = patterns[:5]

    supporter = [p[0][0] for p in patterns]
    supporters[conf] = supporter

# Get paper counts of supporters in each year
yearly_counts = dict()
for conf in sorted_confs:
    yearly_counts[conf] = dict()
    conf_data = get_filtered_data(data, key='conference', values=[conf])

    for author in supporters[conf]:
        yearly_counts[conf][author] = []
        author_data = get_filtered_data(conf_data, key='author', values=[author])

        for year in sorted_years:
            count = get_counts(author_data, key='year', values=[year])
            yearly_counts[conf][author].append(count)

# Print supporters for each conference
print_divider('=', 'Question 1.1: Supporter')
for key, value in supporters.items():
    print('{}: {}'.format(key, value))

# Print paper counts of supporters in each year
print_divider('=', 'Question 1.1: Yearly Counts')
for key1, value1 in yearly_counts.items():
    print('{}:'.format(key1))
    for key2, value2 in value1.items():
        print('{}: {}'.format(key2, value2))
    print(' ')

# Print inactive supporters
print_divider('=', 'Question 1.1: Inactive')
for key1, value1 in yearly_counts.items():
    print('{}:'.format(key1))
    for key2, value2 in value1.items():
        if sum(value2[-2:]) <= 0.1 * sum(value2):
            print('{}'.format(key2))
    print(' ')

####################### Question 1.2 ##############################

# Find frequent teams
baskets = get_baskets(data, 'author')
patterns = pyfpgrowth.find_frequent_patterns(baskets, 10)
# Get only teams of more than 3 person
patterns = dict((k, v) for k, v in patterns.items() if len(k) >= 3)
# Remove duplicate
remove_unclosed_items(patterns)
# Sort by frequency
patterns = get_sorted_patterns(patterns)
teams = [list(p[0]) for p in patterns]

# Print frequent teams
print_divider('=', 'Question 1.2: Frequent Teams')
for entry in patterns:
    print('{}: {}'.format(entry[0], entry[1]))

####################### Question 2.1 ##############################

# Get keywords using RAKE
titles = get_values(data, 'title')
titles = sorted(titles)

stopwords = nltk.corpus.stopwords.words('english')
extra_stopwords = ['using', 'via', 'without',
                   'towards', 'toward', 'based']
stopwords += extra_stopwords

extractor = Rake(stopwords=stopwords)
extractor.extract_keywords_from_sentences(titles)

keywords = extractor.get_ranked_phrases()
# Only using keywords with less than 5 words
keywords = [k for k in keywords if len(k.split()) <= 5]
# Get 20000 keywords
keywords = keywords[:20000]

# Get keyword of each team's research
print_divider('=', 'Question 2.1: Team Insterests')
for team in teams:
    team_data = get_team_data(data, team)
    team_titles = get_values(team_data, 'title')
    team_titles = sorted(team_titles)
    team_freq = get_ranked_keyword_frequency(team_titles, keywords)

    print_divider('-', str(team))
    for entry in team_freq:
        print('{}: {}'.format(entry[0], entry[1]))

####################### Question 2.2 ##############################

# 2007 - 2011
first_half_years = sorted_years[:len(sorted_years) // 2]
# 2012 - 2017
second_half_years = sorted_years[len(sorted_years) // 2:]

print_divider('=', 'Question 2.2: Team Insterests in Two Periods')
for team in teams:
    team_data = get_team_data(data, team)

    # Fisrt half
    team_first_data = get_filtered_data(team_data, key='year', values=first_half_years)
    team_first_titles = get_values(team_first_data, 'title')
    team_first_titles = sorted(team_first_titles)
    team_first_freq = get_ranked_keyword_frequency(team_first_titles, keywords)
    # Second half
    team_second_data = get_filtered_data(team_data, key='year', values=second_half_years)
    team_second_titles = get_values(team_second_data, 'title')
    team_second_titles = sorted(team_second_titles)
    team_second_freq = get_ranked_keyword_frequency(team_second_titles, keywords)

    print_divider('-', str(team))
    print('\n2007 - 2011: {} papers published\n'.format(len(team_first_data)))
    for entry in team_first_freq:
        print('{}: {}'.format(entry[0], entry[1]))
    print('\n2012 - 2017: {} papers published\n'.format(len(team_second_data)))
    for entry in team_second_freq:
        print('{}: {}'.format(entry[0], entry[1]))
