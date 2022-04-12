import re
import csv

from scrapy import Spider, Request
from scrapy.http import TextResponse
from scrapy.linkextractors.lxmlhtml import LxmlLinkExtractor

from newspaper import Article
from scraping.utils import filter_html


class RawHtmlSpider(Spider):
    # name = "raw_html_spider"
    # urls = [ 'http://myhugo.sysang/posts/' ]
    is_shallow = False
    allow_domains = [
            'myhugo.sysang'
        ]

    deny = [
      'sitemap',
      'sitemaps',
      'sitemap\.xml',
      'sitemaps\.xml',
    ]

    def __init__(self):
        super(RawHtmlSpider, self).__init__()

        self.link_extractor = LxmlLinkExtractor(
                unique=True,
                allow_domains=self.allow_domains,
                deny=self.deny,
            )

        self.l1checked = {}
        self.l2checked = {}

    def start_requests(self):

        for url in self.urls:
            yield Request(url=url, callback=self.l1parse)

    def l1parse(self, response):
        # print('[TEST:]', self.link_extractor.extract_links(response))
        if self.is_shallow:
            callback = self.l3parse
        else:
            callback = self.l2parse

        response = self.remove_nagivation_sections(response)
        for link in self.link_extractor.extract_links(response):
            link_url = self.simplify_url(link.url)

            if self.l1checked.get(link_url):
                continue

            self.l1checked[link_url] = True

            yield Request(link.url, callback=callback)

        yield self.construct_output(response)

    def l2parse(self, response: TextResponse):
        response = self.remove_nagivation_sections(response)
        for link in self.link_extractor.extract_links(response):
            link_url = self.simplify_url(link.url)

            if self.l2checked.get(link_url):
                continue

            self.l2checked[link_url] = True

            yield Request(link.url, callback=self.l3parse)

        yield self.construct_output(response)

    def l3parse(self, response):
        yield self.construct_output(response)

    def remove_nagivation_sections(self, response):
        response.selector.xpath('//nav').remove()
        response.selector.xpath('//footer').remove()

        return response

    def construct_output(self, response):

        return {
                'url': self.process_url(response),
                'text': self.process_text(response),
                'original': self.process_original(response),
            }

    # Remove hash symbol
    def simplify_url(self, url):
        p = re.compile('\#.*$')

        return p.sub('', url)

    def parse_response(self, response):
        html_text = response.xpath('//body').get()

        return html_text

    def process_url(self, response):
        return response.url

    def process_text(self, response):
        return filter_html(self.parse_response(response))

    def process_original(self, response):
        return self.parse_response(response)

class RawHtmlAPNews(RawHtmlSpider):
    name = "raw_html_apnews"
    is_shallow = False

    allow_domains = [
            'apnews.com',
        ]

    urls = [
          'https://apnews.com'
        ]

class RawHtmlReuters(RawHtmlSpider):
    name = "raw_html_reuters"
    is_shallow = False

    allow_domains = [
            'reuters.com',
        ]

    urls = [
          'https://www.reuters.com/'
        ]

class RawHtmlChannalNewsAsiaComAsia(RawHtmlSpider):
    name = "raw_html_cna_asia"
    is_shallow = False

    allow_domains = [
            'channelnewsasia.com',
            'cnalifestyle.channelnewsasia.com',
        ]

    urls = [
            'https://www.channelnewsasia.com/asia',
            'https://www.channelnewsasia.com/world',
            'https://www.channelnewsasia.com/business',
            'https://cnalifestyle.channelnewsasia.com/entertainment',
            'https://cnalifestyle.channelnewsasia.com/women',
            'https://cnalifestyle.channelnewsasia.com/wellness',
            'https://cnalifestyle.channelnewsasia.com/living',
            'https://cnalifestyle.channelnewsasia.com/dining',
            'https://cnalifestyle.channelnewsasia.com/travel',
        ]

    deny = [
            'www\.channelnewsasia\.com\/?$',
            'cnalifestyle\.channelnewsasia\.com\/?$',
        ] + RawHtmlSpider.deny


class RawHtmlWikipediaOrgWikiHouse(RawHtmlSpider):
    name = "raw_html_wikipeida_house"
    is_shallow = False
    allow_domains = [
            'en.wikipedia.org',
        ]
    urls = [
            'https://en.wikipedia.org/wiki/House',
        ]
    deny = [
            'en\.wikipedia\.com\/?$',
        ] + RawHtmlSpider.deny

class RawHtmlWikipediaOrgBuilding(RawHtmlSpider):
    name = "raw_html_wikipeida_building"
    is_shallow = False
    allow_domains = [
            'en.wikipedia.org',
        ]
    urls = [
            'https://en.wikipedia.org/wiki/Building',
        ]
    deny = [
            'en\.wikipedia\.com\/?$',
        ] + RawHtmlSpider.deny

class RawHtmlTheGuardianComLifestyle(RawHtmlSpider):
    name = "raw_html_the_guardian"
    is_shallow = False
    allow_domains = [
            'theguardian.com',
        ]
    urls = [
            'https://www.theguardian.com/uk/commentisfree',
            'https://www.theguardian.com/international',
            'https://www.theguardian.com/uk/money',
            'https://www.theguardian.com/uk/travel',
            'https://www.theguardian.com/lifeandstyle/family',
            'https://www.theguardian.com/lifeandstyle/men',
            'https://www.theguardian.com/lifeandstyle/women',
            'https://www.theguardian.com/lifeandstyle/home-and-garden',
            'https://www.theguardian.com/lifeandstyle/health-and-wellbeing',
            'https://www.theguardian.com/lifeandstyle/love-and-sex',
            'https://www.theguardian.com/food',
            'https://www.theguardian.com/fashion',
        ]
    deny = [
            'www\.theguardian\.com\/?$',
        ] + RawHtmlSpider.deny

class RawHtmlNewsweek(RawHtmlSpider):
    name = "raw_html_newsweek"
    is_shallow = False
    allow_domains = [
            'newsweek.com',
        ]
    urls = [
          'https://www.newsweek.com',
        ]

class RawHtmlPolitico(RawHtmlSpider):
    name = "raw_html_politico"
    is_shallow = False
    allow_domains = [
            'politico.com',
        ]
    urls = [
          'https://www.politico.com/',
        ]

class RawHtmlTelegraph(RawHtmlSpider):
    name = "raw_html_telegraph"
    is_shallow = False
    allow_domains = [
            'telegraph.co.uk',
        ]
    urls = [
          'https://www.telegraph.co.uk/',
        ]

class RawHtmlNYPost(RawHtmlSpider):
    name = "raw_html_nypost"
    is_shallow = False
    allow_domains = [
            'nypost.com',
        ]
    urls = [
          'https://nypost.com/',
        ]

class RawHtmlSMH(RawHtmlSpider):
    name = "raw_html_smh"
    is_shallow = False
    allow_domains = [
            'smh.com.au',
        ]
    urls = [
          'https://www.smh.com.au/',
        ]

class RawHtmlWaPo(RawHtmlSpider):
    name = "raw_html_wapo"
    is_shallow = False
    allow_domains = [
            'washingtonpost.com',
        ]
    urls = [
          'https://www.washingtonpost.com',
        ]

class RawHtmlFoxBusinessCom(RawHtmlSpider):
    name = "raw_html_foxnews"
    is_shallow = False
    allow_domains = [
            'www.foxbusiness.com',
            'foxbusiness.com',
            'foxnews.com',
        ]
    urls = [
            'https://www.foxbusiness.com/economy',
            'https://www.foxbusiness.com/markets',
            'https://www.foxbusiness.com/real-estate',
            'https://www.foxbusiness.com/technology',
            'https://www.foxnews.com/politics',
        ]
    deny = [
            'www\.foxbusiness\.com\/?$',
        ] + RawHtmlSpider.deny

class RawHtmlSpychologytoday(RawHtmlSpider):
    name = "raw_html_psychologytoday"
    is_shallow = False
    allow_domains = [
            'psychologytoday.com',
        ]
    urls = [
            'https://www.psychologytoday.com/intl',
        ]

class RawHtmlMiscellaneous(RawHtmlSpider):
    name = "raw_html_miscelleneous"
    is_shallow = True

    deny = [
            'sitemap',
            'sitemaps',
            'sitemap\.xml',
            'sitemaps\.xml',
        ]

    allow_domains = [
            'www.goal.com',
            'www.fandom.com',
            'theoceancleanup.com',
            'www.wired.com',
            'corporate.target.com',
            'www.pcworld.com',
            'kidshealth.org',
            'designformankind.com',
            'www.caranddriver.com',
            'www.sciencedaily.com',
        ]
    urls = [
            'https://www.goal.com/en/news/1',
            'https://www.fandom.com/articles/ghostbusters-afterlife-past-future-cast',
            'https://theoceancleanup.com/press/',
            'https://www.wired.com/category/ideas/',
            'https://www.wired.com/category/science/',
            'https://corporate.target.com/article/category/lifestyle',
            'https://corporate.target.com/article/category/company',
            'https://corporate.target.com/article/category/guestexperience',
            'https://www.pcworld.com/reviews',
            'https://www.pcworld.com/news',
            'https://kidshealth.org/en/parents/general/',
            'https://designformankind.com/blog/',
            'https://www.caranddriver.com/news/',
            'https://www.sciencedaily.com/news/top/health/',
        ]


class RawHtmlMiscellaneous2(RawHtmlSpider):
    name = "raw_html_miscelleneous2"
    is_shallow = True

    deny = [
            'sitemap',
            'sitemaps',
            'sitemap\.xml',
            'sitemaps\.xml',
        ]

    allow_domains = [
                'cupofjo.com',
                'camillestyles.com',
                'theblondesalad.com',
                'witanddelight.com',
                'www.thirteenthoughts.com',
                'freshexchange.com',
                'thestripe.com',
                'quintessenceblog.com',
                'www.consciouslifestylemag.com',
                'www.superhitideas.com',
            ]

    urls = [
            'https://cupofjo.com/category/style/beauty/',
            'https://cupofjo.com/category/travel/city-guide-series/',
            'https://cupofjo.com/2021/06/fire-island-house-tour/',
            'https://camillestyles.com/category/food/recipes/breakfast/',
            'https://camillestyles.com/category/wellness/career/',
            'https://theblondesalad.com/en/fashion/',
            'https://theblondesalad.com/en/people/',
            'https://witanddelight.com/category/interiors-decor/',
            'https://www.thirteenthoughts.com/category/photography-2/',
            'https://freshexchange.com/gardening/',
            'https://thestripe.com/category/travel/',
            'https://quintessenceblog.com/category/learning/',
            'https://quintessenceblog.com/category/history/',
            'https://www.consciouslifestylemag.com/category/mindfulness/',
            'https://www.superhitideas.com/home-decor/',
        ]


class RawHtmlMiscellaneous3(RawHtmlSpider):
    name = "raw_html_miscelleneous3"
    is_shallow = True

    deny = [
            'sitemap',
            'sitemaps',
            'sitemap\.xml',
            'sitemaps\.xml',
        ]

    allow_domains = [
          'www.nybooks.com',
          'martinfowler.com',
          'zenexmachina.com',
          'www.information-age.com',
          'pragprog.com',
          'blog.softtek.com',
          'taazaa.com',
          'www.rte.ie',
          'www.secsports.com',
          'www.ballysports.com',
          'www.abc.net.au'
          'www.sportingkc.com',
          'www.theatlantic.com',
          'historyforatheists.com',
          'nymag.com',
          'www.thecut.com',
          'www.vulture.com',
          'wildhorseeducation.org',
          'www.agribusinessglobal.com',
          'www.northqueenslandregister.com.au',
          'www.usda.gov',
          'www.infobip.com',
          'www.teradata.com',
          'www.nhsx.nhs.uk',
        ]

    urls = [
          'https://www.nybooks.com/ideas/',
          'https://martinfowler.com/agile.html',
          'https://zenexmachina.com/blog/',
          'https://www.information-age.com/topics/artificial-intelligence/',
          'https://pragprog.com/categories/android-ios-and-mobile/',
          'https://blog.softtek.com/en',
          'https://taazaa.com/insights/nemt-dispatch-software-development/',
          'https://www.rte.ie/sport/basketball/',
          'https://www.secsports.com/sec/news',
          'https://www.ballysports.com/south/news/',
          'https://www.abc.net.au/news/sport/cricket/'
          'https://www.sportingkc.com/news/',
          'https://www.theatlantic.com/culture/',
          'https://historyforatheists.com/',
          'https://nymag.com/intelligencer/',
          'https://www.thecut.com/culture/',
          'https://www.vulture.com/',
          'https://wildhorseeducation.org/',
          'https://www.agribusinessglobal.com/agtech/',
          'https://www.northqueenslandregister.com.au/news/agribusiness/',
          'https://www.usda.gov/media/blog',
          'https://www.infobip.com/blog',
          'https://www.teradata.com/Blogs',
          'https://www.nhsx.nhs.uk/blogs/',
        ]


class RawHtmlMiscellaneous4(RawHtmlSpider):
    name = "raw_html_miscelleneous4"
    is_shallow = True

    deny = [
            'sitemap',
            'sitemaps',
            'sitemap\.xml',
            'sitemaps\.xml',
        ]

    allow_domains = [
          'www.clickondetroit.com',
          'www.freep.com'
          'www.edmontonpolice.ca',
          'www.insurancejournal.com',
          'www.archdaily.com',
          'aas.org',
          'www.britannica.com',
          'unistellaroptics.com',
        ]

    urls = [
          'https://www.clickondetroit.com/news/',
          'https://www.freep.com/news/detroit/'
          'https://www.edmontonpolice.ca/News/MediaReleases.aspx',
          'https://www.insurancejournal.com/news/international/',
          'https://www.archdaily.com/architecture-news?ad_source=jv-header&ad_name=main-menu',
          'https://aas.org/posts/news/2021/11/russian-anti-satellite-test',
          'https://www.britannica.com/browse/Astronomy',
          'https://unistellaroptics.com/blog-2/',
        ]


class RawHtmlMiscellaneous5(RawHtmlSpider):
    name = "raw_html_miscelleneous5" # validation dataset
    is_shallow = True

    deny = [
            'sitemap',
            'sitemaps',
            'sitemap\.xml',
            'sitemaps\.xml',
        ]

    allow_domains = [
          'www.drrogersprize.org',
          'edzardernst.com',
          'pubmed.ncbi.nlm.nih.gov',
          'www.biorxiv.org',
          'connect.biorxiv.org',
          'mesonstars.com',
          'news.mit.edu',
          'www.scientificamerican.com',
          'finance.yahoo.com',
          'quillette.com',
          'siliconangle.com',
        ]

    urls = [
          'https://www.drrogersprize.org/news/',
          'https://edzardernst.com/contents/',
          'https://pubmed.ncbi.nlm.nih.gov/trending/',
          'https://www.biorxiv.org/collection/bioinformatics',
          'https://connect.biorxiv.org/news/',
          'https://mesonstars.com/category/astrophysics/',
          'https://mesonstars.com/category/science/',
          'https://mesonstars.com/category/inteteresting/',
          'https://mesonstars.com/category/history/',
          'https://news.mit.edu/',
          'https://www.scientificamerican.com/mind-and-brain/',
          'https://www.scientificamerican.com/psychology/',
          'https://finance.yahoo.com/topic/stock-market-news/',
          'https://quillette.com/tag/science-tech/',
          'https://quillette.com/tag/education/',
          'https://siliconangle.com/category/emerging-tech/',
        ]


class RawHtmlMiscellaneous6(RawHtmlSpider):
    name = "raw_html_miscelleneous6" # validation dataset
    is_shallow = True

    deny = [
            'sitemap',
            'sitemaps',
            'sitemap\.xml',
            'sitemaps\.xml',
        ]

    allow_domains = [
          '2pml.com',
          'resistance.kiwi',
          'www.barstoolsports.com',
          'www.iwmbuzz.com',
          'www.bitchmedia.org',
          'retailtechinnovationhub.com',
          'www.retailtechnologyreview.com',
          'www.salesforce.com',
        ]

    urls = [
          'https://2pml.com/archives/',
          'https://resistance.kiwi/rk-blog/',
          'https://www.barstoolsports.com/blogs',
          'https://www.iwmbuzz.com/category/lifestyle',
          'https://www.bitchmedia.org/topic/health',
          'https://retailtechinnovationhub.com/startup-stories',
          'https://www.retailtechnologyreview.com/articles/',
          'https://www.salesforce.com/blog/',
        ]


class RawHtmlMiscellaneous7(RawHtmlSpider):
    name = "raw_html_miscelleneous7" # validation dataset
    is_shallow = True

    deny = [
            'sitemap',
            'sitemaps',
            'sitemap\.xml',
            'sitemaps\.xml',
        ]

    allow_domains = [
          'www.nintendolife.com',
          'researchoutreach.org',
          'www.kicktraq.com',
          'archinect.com',
          'worldarchitecture.org',
          'www.3dnatives.com',
          'www.designboom.com',
          'ultimaker.com',
          'dot.la',
        ]

    urls = [
          'https://www.nintendolife.com/news',
          'https://researchoutreach.org/blog/',
          'https://researchoutreach.org/category/articles/physical-sciences/',
          'http://www.kicktraq.com/',
          'https://archinect.com/news',
          'https://worldarchitecture.org/architecture-news/',
          'https://www.3dnatives.com/en/category/news/',
          'https://www.designboom.com/',
          'https://ultimaker.com/',
          'https://dot.la/news/',
        ]

class RawHtmlMiscellaneous8(RawHtmlSpider):
    name = "raw_html_miscelleneous8" # validation dataset
    is_shallow = True

    deny = [
            'sitemap',
            'sitemaps',
            'sitemap\.xml',
            'sitemaps\.xml',
        ]

    allow_domains = [
          'lifeatlaw.com',
          'www.ucl.ac.uk',
          'asrwritings.com',
          'www.brookings.edu',
          'www.brookings.edu',
          'www.brookings.edu',
          'www.brookings.edu',
          'www.brookings.edu',
          'www.brookings.edu',
          'www.euroweeklynews.com',
          'www.euroweeklynews.com',
          'www.euroweeklynews.com',
          'www.laurenyloves.co.uk',
          'www.laurenyloves.co.uk',
          'www.laurenyloves.co.uk',
          'www.gibbonsforohio.com',
          'www.gibbonsforohio.com',
          'www.bestearningcourse.com',
          'cybercashworldwide.com',
          'cybercashworldwide.com',
        ]

    urls = [
          'https://www.ucl.ac.uk/news/home/latest-news',
          'http://lifeatlaw.com/',
          'https://asrwritings.com/work/',
          'https://www.brookings.edu/topic/international-affairs/',
          'https://www.brookings.edu/topic/telecommunications-internet/',
          'https://www.brookings.edu/topic/energy-industry/',
          'https://www.brookings.edu/topic/u-s-economy/',
          'https://www.brookings.edu/topic/financial-institutions/',
          'https://www.brookings.edu/topic/manufacturing/',
          'https://www.euroweeklynews.com/columnists/leapy-lee/',
          'https://www.euroweeklynews.com/author/david_searl/',
          'https://www.euroweeklynews.com/news/spain/',
          'https://www.laurenyloves.co.uk/category/blogging/',
          'https://www.laurenyloves.co.uk/category/blogging/page/2/',
          'https://www.laurenyloves.co.uk/category/blogging/page/2/',
          'https://www.gibbonsforohio.com/news',
          'https://www.gibbonsforohio.com/issues',
          'https://www.bestearningcourse.com/',
          'https://cybercashworldwide.com/category/vpn',
          'https://cybercashworldwide.com/category/product-reviews',
        ]


class RawHtmlMiscellaneous9(RawHtmlSpider):
    name = "raw_html_miscelleneous9"
    is_shallow = True

    deny = [
            'sitemap',
            'sitemaps',
            'sitemap\.xml',
            'sitemaps\.xml',
        ]

    allow_domains = [
            'www.skepdoc.info',
            'www.townsendletter.com',
            'cs.astronomy.com',
            'astronomy.com',
            'unistellaroptics.com',
            'public.nrao.edu',
            'developer.android.com',
            'howtodoandroid.com',
            'android-developers.googleblog.com',
            'www.kaya959.co.za',
            'www.uts.edu.au',
            ]

    urls = [
            # 'https://www.drrogersprize.org/news',  # Duplicated
            'https://www.skepdoc.info/blog/',
            'https://www.skepdoc.info/category/basic-science/',
            'https://www.skepdoc.info/category/acupuncture/',
            'https://www.skepdoc.info/category/alternative-medicine/',
            'https://www.skepdoc.info/category/alzheimers-disease/',
            'https://www.townsendletter.com/newarticles.htm',
            'https://cs.astronomy.com/asy/b/astronomy/default.aspx',
            'https://cs.astronomy.com/asy/b/daves-universe/default.aspx',
            'https://astronomy.com/news',
            'https://astronomy.com/',
            'https://unistellaroptics.com/blog-2',
            'https://public.nrao.edu/news/',
            'https://developer.android.com/training/wearables',
            'https://developer.android.com/things/get-started',
            'https://developer.android.com/training/cars',
            'https://developer.android.com/training/tv',
            'https://howtodoandroid.com/category/recyclerview/',
            'https://howtodoandroid.com/category/jetpack/',
            'https://howtodoandroid.com/category/library/',
            'https://howtodoandroid.com/category/material-design/',
            'https://howtodoandroid.com/category/kotlin/',
            'https://android-developers.googleblog.com/2021/07/',
            'https://android-developers.googleblog.com/2021/08/',
            'https://android-developers.googleblog.com/2021/09/',
            'https://android-developers.googleblog.com/2021/10/',
            'https://android-developers.googleblog.com/2021/11/',
            'https://www.kaya959.co.za/news/',
            'https://www.uts.edu.au/research-and-teaching/research/research-area/future-industries',
            'https://www.uts.edu.au/research-and-teaching/research/research-area/infrastructure',
            'https://www.uts.edu.au/research-and-teaching/research/research-area/health',
        ]

class RawHtmlMiscellaneous10(RawHtmlSpider):
    name = "raw_html_miscelleneous10"
    is_shallow = True

    deny = [
            'sitemap',
            'sitemaps',
            'sitemap\.xml',
            'sitemaps\.xml',
        ]

    allow_domains = [
        'business.yell.com',
        'nowyoureadme.com',
        'grain.org',
        'www.myfrugalfitness.com',
        'onpassive.com',
        'www.coinhighlight.com',
        'rightsindevelopment.org',
      ]

    urls = [
        'https://business.yell.com/knowledge/page/2/',
        'https://business.yell.com/knowledge/page/3/',
        'https://business.yell.com/knowledge/page/4/',
        'https://business.yell.com/knowledge/page/5/',
        'https://nowyoureadme.com/category/world-news/',
        'https://nowyoureadme.com/category/technology/',
        'https://nowyoureadme.com/category/sports-news/',
        'https://nowyoureadme.com/category/science/',
        'https://nowyoureadme.com/category/health/',
        'https://nowyoureadme.com/category/business/',
        'https://grain.org/en/bulletin_board?page=2',
        'https://grain.org/en/bulletin_board?page=3',
        'https://grain.org/en/bulletin_board?page=4',
        'https://grain.org/en/bulletin_board?page=5',
        'https://www.myfrugalfitness.com/p/money-management-frugal-finance.html',
        'https://www.myfrugalfitness.com/p/seo-google-search-engine-optimization.html',
        'https://www.myfrugalfitness.com/p/cryptocurrency-crypto-for-short.html',
        'https://onpassive.com/blog/artificial-intelligence/',
        'https://onpassive.com/blog/marketing/',
        'https://onpassive.com/blog/community/',
        'https://onpassive.com/blog/2021/11/',
        'https://www.coinhighlight.com/category/bitcoin/',
        'https://www.coinhighlight.com/category/cryptocurrency-news/',
        'https://www.coinhighlight.com/category/ethereum/',
        'https://www.coinhighlight.com/category/blockchain/',
        'https://www.coinhighlight.com/category/coinbase/',
        'https://rightsindevelopment.org/news-page/',
        'https://rightsindevelopment.org/blog/',
      ]

class RawHtmlTopMSites(RawHtmlSpider):
    name = "raw_html_topm_sites"
    is_shallow = True

    deny = [
            'sitemap',
            'sitemaps',
            'sitemap\.xml',
            'sitemaps\.xml',
        ]

    @staticmethod
    def read_data(_start_index):
      _length = 50
      _source_path = 'scraping/spiders/top_million_websites(hackertarget-com).csv'

      if 0 == _length:
        return ([], [])

      with open(_source_path, newline='') as f:
        reader = csv.reader(f)
        site_list = list(reader)[_start_index:(_start_index + _length)]

        _allow_domains = [row[1] for row in site_list]
        _urls = ['https://' + row[1] for row in site_list]

        return (_allow_domains, _urls)

    allow_domains = []

    urls = []


class RawHtmlTopMSitesS0X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s0_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(0)


class RawHtmlTopMSitesS50X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s50_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(50)


class RawHtmlTopMSitesS100X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s100_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(100)


class RawHtmlTopMSitesS150X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s150_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(150)


class RawHtmlTopMSitesS200X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s200_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(200)


class RawHtmlTopMSitesS250X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s250_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(250)


class RawHtmlTopMSitesS300X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s300_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(300)

class RawHtmlTopMSitesS350X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s350_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(350)

class RawHtmlTopMSitesS400X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s400_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(400)

class RawHtmlTopMSitesS450X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s450_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(450)

class RawHtmlTopMSitesS500X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s500_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(500)

class RawHtmlTopMSitesS550X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s550_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(550)

class RawHtmlTopMSitesS600X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s600_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(600)

class RawHtmlTopMSitesS650X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s650_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(650)

class RawHtmlTopMSitesS700X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s700_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(700)

class RawHtmlTopMSitesS750X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s750_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(750)

class RawHtmlTopMSitesS800X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s800_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(800)

class RawHtmlTopMSitesS850X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s850_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(850)

class RawHtmlTopMSitesS900X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s900_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(900)

class RawHtmlTopMSitesS950X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s950_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(950)

class RawHtmlTopMSitesS1000X50(RawHtmlTopMSites):
    name = "raw_html_topm_sites_s1000_x50"

    allow_domains, urls = RawHtmlTopMSites.read_data(1000)
