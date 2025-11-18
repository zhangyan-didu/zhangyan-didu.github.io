import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import { useState } from 'react';

import styles from './index.module.css';

const poetryList = [
  "茶烟一缕轻轻扬，搅动兰膏四座香。",
  "寒夜客来茶当酒，竹炉汤沸火初红。",
  "矮纸斜行闲作草，晴窗细乳戏分茶。",
  "茶甘露爽醍醐味，喜受阳和入骨吹。",
  "独携天上小团月，来试人间第二泉。",
  "坐酌泠泠水，看煎瑟瑟尘。",
  "无由持一碗，寄与爱茶人。"
];

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  const [currentPoetry, setCurrentPoetry] = useState<string>('');

  const handleTeaClick = () => {
    const randomIndex = Math.floor(Math.random() * poetryList.length);
    setCurrentPoetry(poetryList[randomIndex]);
  };

  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <button
            className="button button--secondary button--lg"
            onClick={handleTeaClick}
          >
            饮一杯茶
          </button>
        </div>
        {currentPoetry && (
          <div className={styles.poetryDisplay}>
            <p className={styles.poetryText}>{currentPoetry}</p>
          </div>
        )}
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Description will go into a meta tag in <head />">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
