# Aggregated QA 数据集综合报告

本报告合并数据分布、context token 规模和代表性样例。样例中的 context 仅截断展示，原文保存在 aggregated 目录。

## 统计口径

- context：按完整 context 字符串聚合后的记录。
- QA：所有 qa_pairs 的总数。
- token：本地 Qwen3.5 tokenizer 抽样估计；唯一 token 按 context 去重，QA 展开 token 会重复计算。
- split context 相加可能大于总 context，因为同一 context 可跨 split 出现。

## 总览


| 数据集      |     context |            QA | 唯一 context tokens | QA 展开 tokens | 平均 tokens/context | split（context / QA）                                                                                                                                                                        |
| ----------- | ----------: | ------------: | ------------------: | -------------: | ------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| coqa        |       7,070 |       116,630 |               2.51M |         41.47M |                 355 | train: 6,605 / 108,647; validation: 499 / 7,983                                                                                                                                              |
| drop        |       6,108 |        86,935 |               1.88M |         25.60M |                 308 | train: 5,541 / 77,400; validation: 579 / 9,535                                                                                                                                               |
| duorc       |      14,240 |       187,213 |              14.63M |        227.87M |               1,028 | paraphraserc_dev: 1,133 / 15,591; paraphraserc_test: 1,207 / 15,857; paraphraserc_train: 5,133 / 69,524; selfrc_dev: 983 / 12,961; selfrc_test: 1,011 / 12,559; selfrc_train: 4,799 / 60,721 |
| ms_marco    |     410,733 |       419,741 |             101.70M |        102.39M |                 248 | test: 110,707 / 110,742; train: 191,708 / 197,859; validation: 110,642 / 111,140                                                                                                             |
| narrativeqa |       1,572 |        46,765 |             115.92M |       3442.37M |              73,739 | test: 355 / 10,557; train: 1,102 / 32,747; validation: 115 / 3,461                                                                                                                           |
| pwc         |      17,606 |       259,710 |               8.41M |        124.08M |                 478 | test: 1,224 / 18,146; train: 16,382 / 241,564                                                                                                                                                |
| qasper      |       1,585 |         5,049 |               8.39M |         26.53M |               5,296 | test: 416 / 1,451; train: 888 / 2,593; validation: 281 / 1,005                                                                                                                               |
| quac        |       7,843 |        90,922 |               4.48M |         50.39M |                 572 | train: 6,843 / 83,568; validation: 1,000 / 7,354                                                                                                                                             |
| race        |       2,707 |        14,122 |               1.24M |          6.99M |                 457 | dev: 136 / 712; test: 135 / 708; train: 2,436 / 12,702                                                                                                                                       |
| scidqa      |         727 |         3,252 |              17.60M |         82.80M |              24,215 | multidoc: 191 / 315; train: 727 / 2,937                                                                                                                                                      |
| squad       |      21,097 |       240,361 |               3.37M |         39.36M |                 160 | train-v1.1: 18,891 / 87,599; train-v2.0: 19,029 / 130,319; validation-v1.1: 2,067 / 10,570; validation-v2.0: 1,204 / 11,873                                                                  |
| **合计**    | **491,288** | **1,470,700** |      **约 280.14M** |   **约 4.17B** |                  — | —                                                                                                                                                                                           |

Context token 长度分布（用于生成 batching）

以下统计针对 `aggregated/*/contexts.jsonl` 中的 491,288 条 context，使用本地 Qwen3.5-9B tokenizer，并将 `qa_gen/fact_extract_prompt.txt` 中的固定提示词计入输入长度。p50、p90、p95 分别表示第 50、90、95 百分位；最大值用于识别长尾样本。


| 数据集       |  context 数 | p50 tokens | p90 tokens | p95 tokens | 最大 tokens |
| ------------ | ----------: | ---------: | ---------: | ---------: | ----------: |
| coqa         |       7,070 |        923 |      1,004 |      1,038 |       1,891 |
| drop         |       6,108 |        840 |      1,026 |      1,158 |       2,870 |
| duorc        |      14,240 |      1,385 |      2,525 |      3,330 |      16,041 |
| ms_marco     |     410,733 |        670 |      1,280 |      1,360 |       3,035 |
| narrativeqa  |       1,572 |     45,445 |    167,930 |    239,319 |     506,738 |
| pwc          |      17,606 |      1,044 |      1,136 |      1,152 |       1,487 |
| qasper       |       1,585 |      5,518 |      8,569 |     10,197 |      36,469 |
| quac         |       7,843 |      1,086 |      1,290 |      1,380 |       2,982 |
| race         |       2,707 |        996 |      1,255 |      1,399 |       2,040 |
| scidqa       |         727 |     19,493 |     45,634 |     58,546 |     119,051 |
| squad        |      21,097 |        716 |        816 |        857 |       1,546 |
| **全部混合** | **491,288** |    **688** |  **1,296** |  **1,423** | **506,738** |

整体分布具有明显长尾：大多数 context 的长度约为 0.7k–1.4k tokens，但 NarrativeQA、SciDQA 和 QASPER 含有大量长文档。NarrativeQA 的 p90 已达到 167,930 tokens，SciDQA 的 p95 为 58,546 tokens；这些样本不能在常见 8k/32k 上下文窗口中直接生成，应先按 story/段落切块或采用滑动窗口。

按原始文件顺序组成 batch 时，padding 开销较大。大规模生成应按 token 长度进行 sortish batching 或分桶，并结合动态 token budget（约束 `batch_size × (输入长度 + max_new_tokens)`）。实际实现不必全局排序：可先随机打乱，再在 1,000–5,000 条样本的窗口内按长度排序，同时保留原始 `id` 以恢复结果顺序。对失败样本 retry 时也应重新按长度分桶。

## 重点注意事项

- MS MARCO：v1.1/v2.1 共用 aggregated/ms_marco；跨版本重复 context 3,910 个，涉及 8,341 条 QA。
- SQuAD：v1.1/v2.0 大量复用 context；v2.0 不可回答问题不能删除。
- SciQA：multidoc 是 train 子集，不应相加。
- DuoRC：SelfRC 和 ParaphraseRC 是不同设置；no_answer 需要保留。
- QASPER/SciQA：evidence 是重要评估字段。
- NarrativeQA：context 极长，应按 story 划分。

各数据集特点与代表性样例

### coqa

**特点：** 多轮对话；同一 story context 被多个 turn 复用。

**分布：** train: 6,605 / 108,647; validation: 499 / 7,983。
**特殊标注：** 空答案 0；no_answer 0；evidence QA 0。

#### 样例 1（普通样例）

Context（截断）： Roman Britain ( or, later, "", "the Britains") was the area of the island of Great Britain that was governed by the Roman Empire, from 43 to 410 AD. Julius Caesar invaded Britain in 55 and 54 BC as part of his Gallic Wars. The Britons had been overrun or culturally assimilated by other Celtic tribes during the British Iron Age and had been aiding Caesar's enemies. He received tribute, installed a friendly king over the Trinovantes, and returned to Gaul. Planned invasions under Augustus were called off in 34, 27, and 25 BC. In 40 AD, Caligula assembled 200,000 men at the Channel, only to have t…

Question： What did Caesar invade?
Answer： Britain

#### 样例 2（普通样例）

Context（截断）： New York (CNN) -- A self-described "ex-madam" who claims she supplied fellow city comptroller candidate Eliot Spitzer with escorts several years ago is facing charges of illegally distributing prescription drugs, authorities said. Kristin Davis, 38, was arrested on Monday night and charged with selling Adderall, Xanax and other drugs. She's also accused of orchestrating the sale of approximately 180 oxycodone pills for cash. The candidate was released Tuesday on $100,000 bail, with a preliminary hearing scheduled for September 5. Prosecutors said she will have strict pretrial supervision. "Pre…

Question： Is Allison Davis the campaign manager?
Answer： no

#### 样例 3（普通样例）

Context（截断）： Colleges taking another look at value of merit-based aid Good grades and high tests scores still matter--a lot--to many colleges as they award financial aid. But with low-income students projected to make up an ever-larger share of the college-bound population in coming years, some schools are re-examining whether that aid, typically known as "merit aid", is the most effective use of precious institutional dollars. George Washington University in Washington, D.C., for example, said last week that it would cut the value of its average merit scholarships by about one-third and reduce the number…

Question： where is the University named after the first president of the US?
Answer： Washington, D.C.

### drop

**特点：** 离散推理；答案不一定是原文 span。

**分布：** train: 5,541 / 77,400; validation: 579 / 9,535。
**特殊标注：** 空答案 0；no_answer 0；evidence QA 0。

#### 样例 1（普通样例）

Context（截断）： Coming off their win over the Seahawks the 49ers played against the Chargers at Qualcomm Stadium on Thursday Night. The 49ers fell behind 2 minutes into the game with QB Philip Rivers throwing a 58-yard TD pass to WR Vincent Jackson, followed in the second quarter by kicker Nate Kaeding nailing a 25-yard field goal, then with Rivers finding Jackson on an 11-yard touchdown pass. In the third quarter the 49ers had a touchdown from the second-half kickoff return but was declined because of a face-masking penalty enforced on them. Following that, they struggled further with FB Mike Tolbert getting…

Question： Who caught the longest touchdown reception of the game?
Answer： Vincent Jackson

#### 样例 2（普通样例）

Context（截断）： As of the 2010 United States Census, there were 1,951,269 people, 715,365 households, and 467,916 families residing in the county. The population density was . There were 840,343 housing units at an average density of . The racial makeup of the county was 60.9% white, 10.5% black or African American, 8.7% Asian, 0.7% Pacific islander, 0.7% American Indian, 13.5% from other races, and 5.1% from two or more races. Those of Hispanic or Latino origin made up 29.1% of the population. In terms of ancestry, 11.7% were Germans, 9.1% were Irish people, 7.6% were English people, 6.3% were Italians, and…

Question： How many more people were there than households?
Answer： 1235904

#### 样例 3（普通样例）

Context（截断）： After snapping their three-game losing streak against the Seattle Seahawks the weekend before, the Packers returned home to face their first AFC opponent of the season, the 2007 AFC South Champion Indianapolis Colts. The game started with the Packers in possession and the Packers started off the game with a nice 14-yard run by RB Ryan Grant for a first down. On the first third down of the drive, Aaron Rodgers completed his first pass of the day to TE Donald Lee for an 11-yard gain. Colts CB Marlin Jackson was flagged for unnecessary roughness after the play and the Colts were penalized 15&#160…

Question： How many yards was the penalty Jennings received for defensive holding?
Answer： 5

### duorc

**特点：** SelfRC 与 ParaphraseRC 是不同设置；含 no_answer。

**分布：** paraphraserc_dev: 1,133 / 15,591; paraphraserc_test: 1,207 / 15,857; paraphraserc_train: 5,133 / 69,524; selfrc_dev: 983 / 12,961; selfrc_test: 1,011 / 12,559; selfrc_train: 4,799 / 60,721。
**特殊标注：** 空答案 16,547；no_answer 16,547；evidence QA 0。

#### 样例 1（普通样例）

Context（截断）： Set in London, circa 1900, George and Mary Darling's preparations to attend a party are disrupted by the antics of their boys, John and Michael, acting out a story about Peter Pan and the pirates, told to them by their older sister, Wendy. Their father, who is fed up with the stories that have made his children less practical, angrily declares that Wendy has gotten too old to continue staying in the nursery with them. That night, they are visited in the nursery by Peter Pan himself, who teaches them to fly with the help of his pixie friend, Tinker Bell, and takes them with him to the island of…

Question： Where does Peter Pan live?
Answer： Neverland；Never Land

#### 样例 2（普通样例）

Context（截断）： It is winter in Teheran. Lateef is 17. He works at a building construction site managed by MEMAR, the site foreman. Lateef's job is to serve tea and prepare food for the workers with whom he is always quarrelling. The workers come from all parts of Iran, particularly from Iranian Azerbaijan (Azeris are referred as "Turks" in the film). Some workers are Afghan refugees from war-torn Afghanistan. They have no identity cards and are employed illegally as cheap labour. When the labour inspectors show up, the Afghan workers must hide. As the story starts, an Afghan worker, NAJAF, falls from the bui…

Question： Rahmat does not express herself at all how?
Answer： because her real name is Baran

#### 样例 3（普通样例）

Context（截断）： The night had brought little relief from the heat, and at dawn a hot gust of wind blows across the colorless sea. The KNIGHT, Antonius Block, lies prostrate on some spruce branches spread over the fine sand. His eyes are wide-open and bloodshot from lack of sleep.Nearby his squire JONS is snoring loudly. He has fallen asleep where he collapsed, at the edge of the forest among the wind-gnarled fir trees. His open mouth gapes towards the dawn, and unearthly sounds come from his throat. At the sudden gust of wind, the horses stir, stretching their parched muzzles towards the sea. They are as thin…

Question： who is the one that reappears?
Answer： Death

### ms_marco

**特点：** v1.1/v2.1 已合并；需保留 source_version；有大量空答案。

**分布：** test: 110,707 / 110,742; train: 191,708 / 197,859; validation: 110,642 / 111,140。
**特殊标注：** 空答案 103,819；no_answer 0；evidence QA 0。

#### 样例 1（version=2.1）

Context（截断）： A manager title in the workplace can cover a realm of duties, most of them supervisory in nature. In larger corporations, you may find tiers of management levels, each with specific duties. But in a small business, the manager is often a jack-of-all-trades. Though he may oversee aspects of the business, his responsibilities may be hands-on as well. Managers are responsible for staffing the business. In a small business, this includes creating job descriptions, running advertisements for open positions, reviewing resumes and applications, interviewing prospective employees, hiring and firing.

Question： what are a managers responsibilities
Answer： It is responsible for staffing the business.

#### 样例 2（version=2.1）

Context（截断）： 1 Upload failed. 2 We are experiencing some problems, please try again. 3 You can only upload files of type PNG, JPG, or JPEG. 4 You can only upload files of type 3GP, 3GPP, MP4, MOV, AVI, MPG, MPEG, or RM. 5 You can only upload photos smaller than 5 MB. 6 You can only upload videos smaller than 600MB.

Question： how long to wait after drinking cold liquid to obtain oral temp
Answer： No Answer Present.

#### 样例 3（empty_answer, version=2.1）

Context（截断）： As plants grow, their roots can break rocks apart. Erosion is a natural process that causes rock to change, break down and crumble. Acid rain causes mechanical weathering. Chemical weathering takes place fastest in a hot. dry climate. Plant roots produce carbonic acid that reacts with minerals in a rock to weaken it Sandstone weathers into a clayey soil As time passes, a soil becomes more like the rock It came from. The Rock Cycle According to the Law of Conservation of Mass, matter cannot be created or destroyed. It only changes form. The same is true about rocks! The Law of Conservation The…

Question： what is the natural process that causes rock to break and crumble calle
Answer： （空答案）

### narrativeqa

**特点：** 长篇故事/摘要 QA；context 极长，应按 story 隔离。

**分布：** test: 355 / 10,557; train: 1,102 / 32,747; validation: 115 / 3,461。
**特殊标注：** 空答案 0；no_answer 0；evidence QA 0。

#### 样例 1（普通样例）

Context（截断）： By Frederik Pohl and C. M. Kornbluth _THE SPACE MERCHANTS_ _SEARCH THE SKY_ ------------------------------------------------------------------------ SEARCH THE SKY by Frederik Pohl and C. M. Kornbluth BALLANTINE BOOKS · NEW YORK ------------------------------------------------------------------------ COPYRIGHT, 1954, BY FREDERIK POHL AND C. M. KORNBLUTH LIBRARY OF CONGRESS CATALOGUE CARD NO. 54-6478 PRINTED IN THE UNITED STATES OF AMERICA BALLANTINE BOOKS, INC. 404 Fifth Avenue, New York 18, N. Y. ------------------------------------ TRANSCRIBER'S NOTE Extensive research did not uncover any ev…

Question： What is Ross sent to do?
Answer： He is sent to find out what has happened to the interstellar colonies.；Discover the state of interstellar colonies

#### 样例 2（普通样例）

Context（截断）： Transcribed form the 1911 W. Foulsham & Co. Ltd. edition by David Price, email ccx074@coventry.ac.uk THE LAIR OF THE WHITE WORM To my friend Bertha Nicoll with affectionate esteem. CHAPTER I--ADAM SALTON ARRIVES Adam Salton sauntered into the Empire Club, Sydney, and found awaiting him a letter from his grand-uncle. He had first heard from the old gentleman less than a year before, when Richard Salton had claimed kinship, stating that he had been unable to write earlier, as he had found it very difficult to trace his grand-nephew's address. Adam was delighted and replied cordially; he had ofte…

Question： Where is Adam Salton originally from?
Answer： He is from Australia.；Australia.

#### 样例 3（普通样例）

Context（截断）： [Illustration] ANNA KARENINA by Leo Tolstoy Translated by Constance Garnett Contents PART ONE PART TWO PART THREE PART FOUR PART FIVE PART SIX PART SEVEN PART EIGHT PART ONE Chapter 1 Happy families are all alike; every unhappy family is unhappy in its own way. Everything was in confusion in the Oblonskys’ house. The wife had discovered that the husband was carrying on an intrigue with a French girl, who had been a governess in their family, and she had announced to her husband that she could not go on living in the same house with him. This position of affairs had now lasted three days, and n…

Question： Who does Couny Vronsky have an affair with?
Answer： Anna Karenina；Anna Karenina

### pwc

**特点：** 指令型 context-to-text；包含问答、摘要、抽取、改写和解释。

**分布：** test: 1,224 / 18,146; train: 16,382 / 241,564。
**特殊标注：** 空答案 0；no_answer 0；evidence QA 0。

#### 样例 1（普通样例）

Context（截断）： The vast majority of the federal cybersecurity workforce is older than 40, an issue that could eventually lead to a personnel shortage in the field, according to a new report. The 2012 Information Technology Workforce Assessment for Cybersecurity, released Wednesday, found that nearly 80 percent of federal cybersecurity workers are over the age of 40, with most being closer to the retirement age threshold. Only 5 percent of the federal cyber workforce is 30 years of age or younger, the study found. The report, which is based on an anonymous survey of nearly 23,000 cyber workers across 52 depar…

Question： Explain the potential risk of having an aging cybersecurity workforce in the federal government.
Answer： The potential risk of having an aging cybersecurity workforce in the federal government is that it could lead to a manpower shortage in the field, as a significant number of these professionals may retire in the near future, leaving a gap in the workforce that needs to be filled.

#### 样例 2（普通样例）

Context（截断）： The Philippines faces its toughest challenge yet in Uzbekistan as the Azkals resume their World Cup Qualifying campaign on Sept. 8 at Philippine Sports Stadium in Bocaue, Bulacan. Ranked No. 74 by Fifa, the Uzbeks will also be looking for their first victory in the qualifying tournament against the Azkals, who are running second to North Korea on goal difference in Group H after wins over Bahrain (2-1) and Yemen (2-0). ADVERTISEMENT Having spent long hours the past two months studying Uzbekistan play, Dooley feels the Central Asians are the strongest team in Group H despite their 2-4 defeat to…

Question： Rephrase the above text
Answer： The Philippines' national football team, the Azkals, is set to face their most formidable opponent yet, Uzbekistan, on September 8th at the Philippine Sports Stadium in Bocaue, Bulacan, as part of their ongoing World Cup Qualifying campaign. Uzbekistan, ranked 74th by FIFA, is also seeking their first win in the qualifiers, while the Azkals are in second place behind North Korea in Group H after defeating Bahrain (2-1) and Yemen (2-0). Coach Dooley has spent considerable time analyzing Uzbekistan's gameplay and believes they are the most challenging team in their group, despite a 2-4 loss to North Korea in June. Nevertheless, Dooley is optimistic about the Azkals' chances, emphasizing the need for each player to give their all. The Azkals will have a warm-up match against the Maldives on September 3 at Rizal Memorial Stadium before facing Uzbekistan. Several international players are expected to join the Azkals' training sessions in the coming days.

#### 样例 3（普通样例）

Context（截断）： We provide a 24/7 service to the UK and beyond. We welcome all related enquiries and all communications are 100% confidential at all times. Contact Us today if you think we can help. We provide a 24/7 service and we welcome all business enquiries. All communications are completely confidential at all times. If you are unable to reach us via phone on 0800 043 6667 or on our alternative number then please leave us a message/email and we will reply to you ASAP. Our Services - More For more detailed information about any of our services please click on the relevant link below and you will be taken…

Question： List the services provided by this company.
Answer： General investigation, Mediation, Surveillance, Tracing persons, Vehicle tracking, Counter surveillance, Background checks, Full company reports, Age and Identity Verification, Drugs and Teenagers assistance, Trace and locate services, Missing People Service, Asset Investigation, Professional Claim, Fraud and Theft Investigation Services.

### qasper

**特点：** 科研论文 QA；evidence 是关键字段，部分问题无答案。

**分布：** test: 416 / 1,451; train: 888 / 2,593; validation: 281 / 1,005。
**特殊标注：** 空答案 410；no_answer 0；evidence QA 5,049。

#### 样例 1（evidence）

Context（截断）： # Analyzing Word Translation of Transformer Layers ## Abstract The Transformer translation model is popular for its effective parallelization and performance. Though a wide range of analysis about the Transformer has been conducted recently, the role of each Transformer layer in translation has not been studied to our knowledge. In this paper, we propose approaches to analyze the translation performed in encoder / decoder layers of the Transformer. Our approaches in general project the representations of an analyzed layer to the pre-trained classifier and measure the word translation accuracy.…

Question： How much is decoding speed increased by increasing encoder and decreasing decoder depth?
Answer： the Transformer with 10 encoder layers and 2 decoder layers is $2.32$ times as fast as the 6-layer Transformer

#### 样例 2（evidence）

Context（截断）： # Question Answering based Clinical Text Structuring Using Pre-trained Language Model ## Abstract Clinical text structuring is a critical and fundamental task for clinical research. Traditional methods such as taskspecific end-to-end models and pipeline models usually suffer from the lack of dataset and error propagation. In this paper, we present a question answering based clinical text structuring (QA-CTS) task to unify different specific tasks and make dataset shareable. A novel model that aims to introduce domain-specific features (e.g., clinical named entity information) into pre-trained…

Question： What data is the language model pretrained on?
Answer： Chinese general corpus

#### 样例 3（evidence）

Context（截断）： # Semi-Supervised Methods for Out-of-Domain Dependency Parsing ## Abstract Dependency parsing is one of the important natural language processing tasks that assigns syntactic trees to texts. Due to the wider availability of dependency corpora and improved parsing and machine learning techniques, parsing accuracies of supervised learning-based systems have been significantly improved. However, due to the nature of supervised learning, those parsing systems highly rely on the manually annotated training corpora. They work reasonably good on the in-domain data but the performance drops significan…

Question： Which English domains do they evaluate on?
Answer： Conll, Weblogs, Newsgroups, Reviews, Answers

### quac

**特点：** 多轮信息寻求对话；问题可能依赖历史 turn。

**分布：** train: 6,843 / 83,568; validation: 1,000 / 7,354。
**特殊标注：** 空答案 0；no_answer 0；evidence QA 0。

#### 样例 1（普通样例）

Context（截断）： Michael Spencer left Flotsam and Jetsam shortly after a U.S. tour in the fall of 1987; his replacement was Troy Gregory. Their second studio album, No Place for Disgrace, was released in May 1988, and includes a cover of Elton John's hit "Saturday Night's Alright for Fighting" for which a music video was shot. The band toured heavily behind No Place for Disgrace throughout 1988 and 1989. They opened for King Diamond in America, and supported Megadeth, Testament and Sanctuary in Europe on the So Far, So Good... So What! tour. The band also played shows with The Crumbsuckers, Fates Warning, Dest…

Question： what happened in 1987?
Answer： Michael Spencer left Flotsam and Jetsam shortly after a U.S. tour in the fall of 1987;

#### 样例 2（普通样例）

Context（截断）： The whereabouts of McDaniel's Oscar are currently unknown. In 1992, Jet magazine reported that Howard University could not find it and alleged that it had disappeared during protests in the 1960s. In 1998, Howard University stated that it could find no written record of the Oscar having arrived at Howard. In 2007, an article in the Huffington Post repeated rumors that the Oscar had been cast into the Potomac River by angry civil rights protesters in the 1960s. The assertion reappeared in the Huffington Post under the same byline in 2009. In 2010, Mo'Nique, the winner of the Academy Award for B…

Question： What are the whereabouts of Hattie's oscar?
Answer： The whereabouts of McDaniel's Oscar are currently unknown.

#### 样例 3（普通样例）

Context（截断）： Olivier-Eugene-Prosper-Charles Messiaen was born December 10, 1908 in Avignon, France, into a literary family. He was the elder of two sons of Cecile Sauvage, a poet, and Pierre Messiaen, a teacher of English who translated the plays of William Shakespeare into French. Messiaen's mother published a sequence of poems, L'ame en bourgeon ("The Budding Soul"), the last chapter of Tandis que la terre tourne ("As the Earth Turns"), which address her unborn son. Messiaen later said this sequence of poems influenced him deeply and he cited it as prophetic of his future artistic career. At the outbreak…

Question： what was the birdsong?
Answer： While he had long been fascinated by birdsong, and birds had made appearances in several of his earlier works

### race

**特点：** 四选一阅读理解；答案是选项而非抽取 span。

**分布：** dev: 136 / 712; test: 135 / 708; train: 2,436 / 12,702。
**特殊标注：** 空答案 0；no_answer 0；evidence QA 0。

#### 样例 1（普通样例）

Context（截断）： Around 100 people have already put down a $10,000 deposit to get a Transition when they go on sale, and those numbers will likely rise after Terrafugia introduces the Transition to the public later this week at the New York Auto Show. But don't expect it to show up in too many driveways. It's expected to cost $279,000.And it won't help if you're stuck in traffic. The car needs a runway. Inventors have been trying to make flying cars since the 1930s, according to Robert Mann, an airline industry expert. But Mann thinks Terrafugia has come closer than anyone to making the flying car a reality. T…

Question： What is the first paragraph mainly about?
Answer： The advantages of flying cars.

#### 样例 2（普通样例）

Context（截断）： Television has transformed politics in the United States by changing the way in which information is disseminated, by altering political campaigns, and by changing citizen's patterns of response to politics. By giving citizens independent access to the candidates, television diminished the role of the political party in the selection of the major party candidates. By centering politics on the person of the candidate, television accelerated the citizen's focus on character rather than issues. Television has altered the forms of political communication as well. The messages on which most of us r…

Question： What is the main point of the passage ?
Answer： Politics in the United States has been significantly changed by television.

#### 样例 3（普通样例）

Context（截断）： （3） He was an old man with a white beard and huge nose and hands. Long before the time during which we will know him, he was a doctor and drove a jaded white horse from house to house through the streets of Winesburg. Later he married a girl who had money. She had been left a large fertile farm when her father died. The girl was quiet, tall, and dark, and to many people she seemed very beautiful. Everyone in Winesburg wondered why she married the doctor. Within a year after the marriage she died. The knuckles of the doctor's hands were extraordinarily large. When the hands were closed they loo…

Question： According to the story Doctor Reefy's life seems very _ .
Answer： eccentric

### scidqa

**特点：** 科学多文档 QA；multidoc 是 train 子集，含 evidence。

**分布：** multidoc: 191 / 315; train: 727 / 2,937。
**特殊标注：** 空答案 0；no_answer 0；evidence QA 3,252。

#### 样例 1（evidence）

Context（截断）： # D2C: Diffusion-Decoding Models for Few-Shot Conditional Generation Abhishek Sinha Department of Computer Science Stanford University a7b23@stanford.edu &Jiaming Song Department of Computer Science Stanford University tsong@cs.stanford.edu &Chenlin Meng Department of Computer Science Stanford University chenlin@cs.stanford.edu &Stefano Ermon Department of Computer Science Stanford University ermon@cs.stanford.edu Equal contribution. ###### Abstract Conditional generative models of high-dimensional images have many applications, but supervision signals from conditions to images can be expensiv…

Question： Discussion about the strength/weakness of the model with varying data diversity would be useful. HINT: Answer needs to be included in the discussion clearly.
Answer： A: It appears that CIFAR-10 and CIFAR-100 are more complex than the face datasets, which may be verified with  with topological data analysis techniques [1].
[1] Khrulkov, V. and Oseledets, I., 2018, July. Geometry score: A method for comparing generative adversarial networks. In International Conference on Machine Learning (pp. 2621-2629). PMLR.

#### 样例 2（evidence）

Context（截断）： # Landmark-RxR: Solving Vision-and-Language Navigation with Fine-Grained Alignment Supervision Anonymous Author(s) Affiliation Address email ###### Abstract In Vision-and-Language Navigation (VLN) task, an agent is asked to navigate inside 3D indoor environments following given instructions. Cross-modal alignment is one of the most critical challenges in VLN because the prediction trajectory needs to match the given instruction accurately. In this paper, we address the cross-modal alignment challenge from a fine-grained perspective. Firstly, to alleviate weak cross-modal alignment supervision…

Question： How does the proposed soft and hard focal-oriented reward relate to the fidelity-based reward from Jain et al, 2019?:
Answer： A: The proposed soft and hard focal-oriented rewards have no direct relationship with the fidelity-based reward [1]. Because the CLS [1] metric is order-invariant, the authors only choose nDTW [2] as the fidelity metric to design the fidelity-oriented reward (model#16 in Table 3) in this paper.
[1] Vihan Jain, Gabriel Magalhaes, Alexander Ku, Ashish Vaswani, Eugene Ie, and Jason Baldridge. Stay on the path: Instruction fidelity in vision-and-language navigation. Association for Computational Linguistics, 2019.
[2] Gabriel Ilharco, Vihan Jain, Alexander Ku, Eugene Ie, and Jason Baldridge. General evaluation for instruction conditioned navigation using dynamic time warping. NeurIPS Visually Grounded Interaction and Language Workshop, 2019.

#### 样例 3（evidence）

Context（截断）： # Flamingo: a Visual Language Model for Few-Shot Learning Anonymous Author(s) Affiliation Address email ###### Abstract Building models that can be rapidly adapted to novel tasks using only a handful of annotated examples is an open challenge for multimodal machine learning research. We introduce Flamingo, a family of Visual Language Models (VLM) with this ability. We propose key architectural innovations to: (i) bridge powerful pretrained vision-only and language-only models, (ii) handle sequences of arbitrarily interleaved visual and textual data, and (iii) seamlessly ingest images or videos…

Question： Does CM3 follow the paper's similar approach?
Reference:
[1] CM3: A Causal Masked Multimodal Model of the Internet, Aghajanyan et al., 2022.
Answer： A: The authors did not intend for the proposed phrasing to suggest that CM3 followed the proposed approach. The authors will modify this description to avoid any ambiguity. The authors will also clarify the architectural differences between the two approaches.

### squad

**特点：** 抽取式 QA；v1.1/v2.0 context 复用，v2.0 含不可回答问题。

**分布：** train-v1.1: 18,891 / 87,599; train-v2.0: 19,029 / 130,319; validation-v1.1: 2,067 / 10,570; validation-v2.0: 1,204 / 11,873。
**特殊标注：** 空答案 49,443；no_answer 0；evidence QA 0。

#### 样例 1（普通样例）

Context（截断）： China is the country with the largest population of Buddhists, approximately 244 million or 18.2% of its total population.[web 1] They are mostly followers of Chinese schools of Mahayana, making this the largest body of Buddhist traditions. Mahayana, also practiced in broader East Asia, is followed by over half of world Buddhists.[web 1]

Question： What country has the largest population of Buddhists?
Answer： China

#### 样例 2（普通样例）

Context（截断）： In 1853, Victoria gave birth to her eighth child, Leopold, with the aid of the new anaesthetic, chloroform. Victoria was so impressed by the relief it gave from the pain of childbirth that she used it again in 1857 at the birth of her ninth and final child, Beatrice, despite opposition from members of the clergy, who considered it against biblical teaching, and members of the medical profession, who thought it dangerous. Victoria may have suffered from post-natal depression after many of her pregnancies. Letters from Albert to Victoria intermittently complain of her loss of self-control. For e…

Question： Who was Victoria's eighth child?
Answer： Leopold

#### 样例 3（普通样例）

Context（截断）： The "freedom to provide services" under TFEU article 56 applies to people who give services "for remuneration", especially commercial or professional activity. For example, in Van Binsbergen v Bestuur van de Bedrijfvereniging voor de Metaalnijverheid a Dutch lawyer moved to Belgium while advising a client in a social security case, and was told he could not continue because Dutch law said only people established in the Netherlands could give legal advice. The Court of Justice held that the freedom to provide services applied, it was directly effective, and the rule was probably unjustified: ha…

Question： The freedom to provide services under TFEU article 56 applies to who?
Answer： to people who give services "for remuneration"；people who give services "for remuneration", especially commercial or professional activity；people who give services "for remuneration"
